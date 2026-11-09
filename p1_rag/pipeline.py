"""Build the system once and answer a questions file (used by the notebook and by graders)."""
import os
import time

import pandas as pd

from common.vlm import VLM
from p1_rag.index import Retriever
from p1_rag.ingest import find_data_dir, ingest
from p1_rag.rag import RAGSystem


def build(data_dir=None, work_dir="work_p1", mode="hybrid", reranker_id=None, vlm=None):
    data_dir = find_data_dir(data_dir)
    vlm = vlm or VLM()
    items, figures = ingest(data_dir, work_dir, vlm)
    retriever = Retriever(items, figures, os.path.join(work_dir, "chroma"), mode=mode, reranker_id=reranker_id)
    return RAGSystem(retriever, vlm), items, figures


def answer_file(system: RAGSystem, questions_path: str, out_csv: str = "submission.csv", with_time=False):
    q = pd.read_csv(questions_path, dtype=str, keep_default_na=False)
    rows, times = [], {}
    for conv_id, turns in q.groupby("conversation_id", sort=False):
        chat = system.new_chat()                      # empty memory per conversation
        for _, r in turns.sort_values("turn", key=lambda t: t.astype(int)).iterrows():
            t0 = time.time()
            a = chat.answer(r["question"])
            times[r["id"]] = time.time() - t0
            rows.append({"id": r["id"], "conversation_id": conv_id, "question_id": r["question_id"],
                         "answer": " ".join(a["answer"].split()),
                         "figure_references": ",".join(a["figure_references"]) or "NONE",
                         "citations": ",".join(a["citations"]) or "NONE", "status": a["status"]})
    out = pd.DataFrame(rows).set_index("id").loc[q["id"]].reset_index()
    out.to_csv(out_csv, index=False)
    return (out, times) if with_time else out
