"""1.1 / 1.2: ChromaDB vector store plus BM25, and hybrid retrieval (RRF)."""
import os

from p1_rag.text_utils import base_label, rrf, tokenize

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Retriever:
    """mode: 'dense' (baseline) | 'hybrid' (dense + BM25). rerank: optional cross-encoder (bonus)."""

    def __init__(self, items, figures, persist_dir, mode="hybrid", reranker_id=None, embedder=None):
        import chromadb
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer
        self.mode = mode
        self.items = {i["id"]: i for i in items}
        self.figures = figures
        self.embedder = embedder or SentenceTransformer(EMBED_MODEL, device="cuda" if _cuda() else "cpu")
        os.makedirs(persist_dir, exist_ok=True)
        client = chromadb.PersistentClient(path=persist_dir)
        try:
            client.delete_collection("kb")
        except Exception:
            pass
        self.col = client.create_collection("kb", metadata={"hnsw:space": "cosine"})
        ids = [i["id"] for i in items]
        for s in range(0, len(items), 128):
            batch = items[s:s + 128]
            emb = self.embedder.encode([b["text"] for b in batch], normalize_embeddings=True, batch_size=32)
            self.col.add(ids=[b["id"] for b in batch], embeddings=emb.tolist(), documents=[b["text"] for b in batch],
                         metadatas=[{k: b[k] for k in ("doc_id", "page", "figure_id", "trust_level", "kind", "untrusted")} for b in batch])
        self.n_indexed = self.col.count()
        trusted = [i for i in items if not i["untrusted"]]
        self.bm25_ids = [i["id"] for i in trusted]
        self.bm25 = BM25Okapi([tokenize(i["text"]) for i in trusted])
        self.reranker = None
        if reranker_id:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(reranker_id, device="cuda" if _cuda() else "cpu")

    def _dense(self, query, n):
        q = self.embedder.encode([QUERY_PREFIX + query], normalize_embeddings=True)
        res = self.col.query(query_embeddings=q.tolist(), n_results=n, where={"untrusted": 0})
        return res["ids"][0]

    def _bm25(self, query, n):
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:n]
        return [self.bm25_ids[i] for i in order if scores[i] > 0]

    def search(self, query: str, k: int = 6) -> list[dict]:
        if self.mode == "dense":
            ids = self._dense(query, k)
        else:
            fused = rrf([self._dense(query, 30), self._bm25(query, 30)])
            ids = [i for i, _ in fused[:max(k, 20 if self.reranker else k)]]
            if self.reranker:
                sc = self.reranker.predict([(query, self.items[i]["text"]) for i in ids])
                ids = [i for _, i in sorted(zip(sc, ids), key=lambda x: -x[0])]
            ids = ids[:k]
        return [self.items[i] for i in ids]

    def figure_lookup(self, labels: list[str]) -> list[dict]:
        """Exact figure/table by label. 'Figure B-5' and 'Figure B-5 (corrected)' are different objects;
        when the plain label is asked for and a corrected sibling exists, both are returned (the revision rule
        needs the replacement), flagged via 'sibling_of'."""
        out, seen = [], set()
        for lab in labels:
            exact = [f for f in self.figures if f["label"] == lab and not f["untrusted"]]
            sib = [] if "(corrected)" in lab else [f for f in self.figures if f["label"] == lab + " (corrected)" and not f["untrusted"]]
            for f in exact + sib:
                key = (f["doc_id"], f["label"], f["page"])
                if key not in seen:
                    seen.add(key)
                    out.append(dict(f, sibling_of=lab if f in sib else ""))
        return sorted(out, key=lambda f: {"high": 0, "medium": 1}.get(f["trust_level"], 2))


def _cuda():
    import torch
    return torch.cuda.is_available()
