"""Dev loop: cached ingestion -> answer Questions.csv -> submission_dev.csv (scored separately by private_eval/score_public.py)."""
import json, os, shutil, sys, time
sys.path.insert(0, os.getcwd())
from common.vlm import VLM, TextLLM
from p1_rag.index import Retriever
from p1_rag.ingest import ingest, find_data_dir
from p1_rag.pipeline import answer_file
from p1_rag.rag import RAGSystem

out = sys.argv[1] if len(sys.argv) > 1 else "submission_dev.csv"
data = find_data_dir()
items, figures = ingest(data, "work_p1", None)
shutil.rmtree("work_p1/chroma_pub", ignore_errors=True)
llm = TextLLM(os.environ["P1_LLM"]) if os.environ.get("P1_LLM", "none") != "none" else None
system = RAGSystem(Retriever(items, figures, "work_p1/chroma_pub", mode="hybrid"), VLM(), llm=llm)
t0 = time.time()
answer_file(system, os.path.join(data, "Questions.csv"), out)
print("answered in", round(time.time() - t0), "s ->", out)
