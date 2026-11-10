"""Fast Part 1 dev loop: reuse cached ingestion (work_p1/items.json, figures.json), score the golden set."""
import json, os, sys, shutil
sys.path.insert(0, os.getcwd())
import pandas as pd
from common.vlm import VLM
from p1_rag.index import Retriever
from p1_rag.rag import RAGSystem
from p1_rag import golden

from p1_rag.ingest import ingest, find_data_dir
if not os.path.exists("work_p1/descriptions.json"):   # seed the cache from the first full run
    old = json.load(open("work_p1/figures.json", encoding="utf-8"))
    json.dump({f"{f['doc_id']}|{f['label']}|{f['page']}": {"description": f["description"], "injection_in_image": f["injection_in_image"]} for f in old}, open("work_p1/descriptions.json", "w", encoding="utf-8"))
items, figures = ingest(find_data_dir(), "work_p1", None)
shutil.rmtree("work_p1/chroma_dev", ignore_errors=True)
vlm = VLM()
llm = None
if os.environ.get("P1_LLM"):
    from common.vlm import TextLLM
    llm = TextLLM(os.environ["P1_LLM"])
final = RAGSystem(Retriever(items, figures, "work_p1/chroma_dev", mode="hybrid"), vlm, llm=llm)
G = golden.load("golden_set.json")
names = sys.argv[1:] or ["final"]
systems = {"final": final}
if "baseline" in names:
    shutil.rmtree("work_p1/chroma_dev_dense", ignore_errors=True)
    systems["baseline"] = RAGSystem(Retriever(items, figures, "work_p1/chroma_dev_dense", mode="dense",
                                              embedder=final.retriever.embedder), vlm, conversation=False)
for name in names:
    recs = golden.run_golden(systems[name], G)
    print(name, golden.summarize(recs))
    gmap = {g["id"]: g for g in G}
    for r in recs:
        if not r["correct"]:
            g = gmap[r["id"]]
            print(f"  WRONG {r['id']} [{r['category']}] status {r['status']} (gold {g['gold_status']}) | pred: {r['answer'][:160]} | cites {r['citations']} figs {r['figure_references']} | gold: {g['gold_answer'][:90]}")
    pd.DataFrame(recs).to_csv(f"golden_results_{name}.csv", index=False)
