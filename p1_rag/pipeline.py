"""Build the system once and answer a questions file (used by the notebook and by graders)."""
import os
import time

import pandas as pd

from common.vlm import VLM
from p1_rag.index import Retriever
from p1_rag.ingest import find_data_dir, ingest
from p1_rag.rag import RAGSystem


ANSWER_LLM = os.environ.get("P1_LLM", "unsloth/Qwen2.5-14B-Instruct-bnb-4bit")   # pre-quantised 4-bit Qwen2.5-14B-Instruct (~9.5 GB); "none" -> VLM answers


DESCRIBE_VLM = os.environ.get("P1_VLM", "Qwen/Qwen2.5-VL-7B-Instruct")   # reads the charts once, at ingestion


def build(data_dir=None, work_dir="work_p1", mode="hybrid", reranker_id=None, vlm=None, answer_llm=ANSWER_LLM):
    """Qwen2.5-VL-7B (4-bit) describes figures at ingestion; Qwen2.5-14B-Instruct (4-bit, ~9.5 GB) writes the answers.
    Both fit a 16 GB T4 because the VLM is unloaded before the answering model is loaded.
    If the 7B VLM cannot be loaded, the 3B VLM is used instead."""
    data_dir = find_data_dir(data_dir)
    if vlm is None:
        vlm = VLM(DESCRIBE_VLM, quant="4bit")
        try:
            items, figures = ingest(data_dir, work_dir, vlm)
        except Exception as e:                          # e.g. quantised checkpoint not loadable -> fall back
            print("describing VLM failed (", type(e).__name__, e, ") - falling back to Qwen2.5-VL-3B")
            vlm.unload()
            vlm = VLM()
            items, figures = ingest(data_dir, work_dir, vlm)
    else:
        items, figures = ingest(data_dir, work_dir, vlm)
    retriever = Retriever(items, figures, os.path.join(work_dir, "chroma"), mode=mode, reranker_id=reranker_id)
    llm = None
    if answer_llm and answer_llm.lower() != "none":
        from common.vlm import TextLLM
        vlm.unload()
        llm = TextLLM(answer_llm)
    return RAGSystem(retriever, vlm, llm=llm), items, figures


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
