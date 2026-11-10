"""Regenerates notebooks/*.ipynb (unexecuted). Run them with Restart & Run All, save outputs, then commit."""
import json
import os

CELL1 = '''# Cell 1 - run information (required)
import subprocess, sys, random, numpy as np, torch
SEED = 266; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
print({"commit": commit, "python": sys.version.split()[0], "torch": torch.__version__,
       "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "seed": SEED})'''

SETUP = '''import os, sys, json, time
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
import pandas as pd
pd.set_option("display.max_colwidth", 120)'''


def nb(cells):
    out = []
    for kind, src in cells:
        c = {"cell_type": kind, "metadata": {}, "source": src.splitlines(True)}
        if kind == "code":
            c.update(execution_count=None, outputs=[])
        out.append(c)
    return {"cells": out, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


P1 = [
 ("markdown", "# Part 1 - Multimodal RAG over a read-only knowledge base\nPipeline: PyMuPDF -> text chunks + figure crops -> Qwen2.5-VL descriptions -> ChromaDB + BM25 -> hybrid (RRF) retrieval -> grounded JSON answers with per-conversation memory.\nData: attach the Kaggle competition data (Document*.pdf, kb_manifest.json, Questions.csv) or set `DATA_DIR`."),
 ("code", CELL1), ("code", SETUP),
 ("markdown", "## 1.1 Ingestion & indexing"),
 ("code", '''from p1_rag.pipeline import build, answer_file
from p1_rag.index import Retriever
from p1_rag.rag import RAGSystem
t0 = time.time()
final, items, figures = build(work_dir="work_p1", mode="hybrid")      # reranker_id="BAAI/bge-reranker-base" for the bonus
print("ingestion+index seconds:", round(time.time() - t0))
fdf = pd.DataFrame(figures)
print(len(items), "indexed items;", len(figures), "figures/tables;", int(fdf.untrusted.sum()), "inside Appendix C (excluded from retrieval)")
fdf[["doc_id", "page", "label", "kind", "trust_level", "untrusted", "cropped", "injection_in_image"]]'''),
 ("markdown", "## 1.2 / 1.3 Retrieval, answering, conversations - sanity check"),
 ("code", '''chat = final.new_chat()
print(chat.answer("What does Figure 1-1 show?"))
print(final.last_retrieved[:5])
n_before = final.retriever.col.count()'''),
 ("markdown", "## 1.4 Golden set (25 questions written by us from the PDFs, committed BEFORE tuning)\nSee `p1_rag/golden.py` for the schema. Save as `golden_set.json` in the repo root."),
 ("code", '''from p1_rag import golden
G = golden.load("golden_set.json")
print("coverage problems:", golden.check_coverage(G) or "none", "| n =", len(G))'''),
 ("code", '''# baseline = dense retrieval only, no follow-up rewriting; final = hybrid + conversation handling
baseline = RAGSystem(Retriever(items, figures, "work_p1/chroma_dense", mode="dense", embedder=final.retriever.embedder),
                     final.vlm, conversation=False, llm=final.llm)   # same answering model; only retrieval/conversation differ
rows = {}
for name, system in [("baseline", baseline), ("final", final)]:
    recs = golden.run_golden(system, G)
    rows[name] = golden.summarize(recs)
    pd.DataFrame(recs).to_csv(f"golden_results_{name}.csv", index=False)
results = pd.DataFrame(rows).T
results'''),
 ("code", '''# Read-only knowledge base: answering must never change the vector store
assert final.retriever.col.count() == n_before, "vector store changed during answering!"
print("vector store unchanged:", n_before, "items")
golden.human_sheet(recs)   # (recs = final-system results from the loop above) both members rate 1-5 in human_eval_p1.csv, then:
# golden.human_summary("human_eval_p1.csv")'''),
 ("markdown", "## C4 safety analysis - untrusted Appendix C, false claims, injections"),
 ("code", '''from p1_rag.text_utils import has_injection
untrusted = [i for i in items if i["untrusted"]]
print("Appendix C items kept out of retrieval:", len(untrusted))
print("figures whose VLM transcript contained instruction-like text:", [f["label"] for f in figures if f["injection_in_image"]])
print("text chunks (trusted) that read like injections before sanitising:", sum(has_injection(i["text"]) for i in items if not i["untrusted"]))'''),
 ("markdown", "## 1.5 Kaggle submission (reads QUESTIONS_PATH, default Questions.csv; no manual steps)"),
 ("code", '''from p1_rag.ingest import find_data_dir
QUESTIONS_PATH = os.environ.get("QUESTIONS_PATH", os.path.join(find_data_dir(), "Questions.csv"))
sub = answer_file(final, QUESTIONS_PATH, "submission.csv")
sub.head()'''),
 ("code", '''import subprocess
print(subprocess.run([sys.executable, os.path.join(find_data_dir(), "validate_submission.py"), "submission.csv", QUESTIONS_PATH], capture_output=True, text=True).stdout)'''),
]

P2 = [
 ("markdown", "# Part 2 - Fine-tuning Stable Diffusion 1.5 with LoRA (rank 4 vs rank 32)"),
 ("code", CELL1), ("code", SETUP),
 ("markdown", "## 2.1 Data preparation"),
 ("code", '''from p2_finetune import data_prep, prompts as P
stats = data_prep.clean_and_split(out_dir="work_p2/data"); print(stats)
test = data_prep.read_split("work_p2/data", "test"); train = data_prep.read_split("work_p2/data", "train")
PROMPTS = P.make_prompts([c for _, c in test], "prompts.json")   # commit prompts.json before training ends
pd.DataFrame(PROMPTS).head(20)'''),
 ("markdown", "Dataset: Naruto BLIP captions (square-cropped to 512x512 - cropping can cut characters). Limitations/biases: (1) one franchise, mostly front-facing characters, so poor generalisation; (2) BLIP captions are short and sometimes wrong (e.g. 'a man with a sword'), which limits prompt alignment."),
 ("markdown", "## 2.2 Two training runs (same data, steps, batch size, lr, seed)"),
 ("code", '''from p2_finetune.train_lora import train
info1 = train("lora_r4", 4, "work_p2/data")
info2 = train("lora_r32", 32, "work_p2/data")
pd.DataFrame([info1, info2]).set_index("run")[["rank", "trainable_params", "learning_rate", "steps", "batch_size", "peak_gpu_mem_gb", "train_seconds", "gpu_hours"]]'''),
 ("code", '''# push to Hugging Face (needs HF_TOKEN in the environment; set repo ids first)
from p2_finetune.hf_upload import push_adapter
push_adapter("work_p2/runs/lora_r4",  "sivasuryachandran/sd15-naruto-lora-r4")
push_adapter("work_p2/runs/lora_r32", "sivasuryachandran/sd15-naruto-lora-r32", results_text="see report")'''),
 ("markdown", "## 2.3 Generation: 20 prompts x 2 seeds x 3 models = 120 images"),
 ("code", '''from p2_finetune.generate import generate_all, SETTINGS
print(SETTINGS)
META = generate_all({"base": None, "run1": "work_p2/runs/lora_r4", "run2": "work_p2/runs/lora_r32"}, PROMPTS)
len(META)'''),
 ("code", '''import matplotlib.pyplot as plt
from PIL import Image
sel = [p for p in PROMPTS if p["type"] == "in_domain"][:4] + [p for p in PROMPTS if p["type"] == "out_of_domain"][:2] + [p for p in PROMPTS if p["type"] == "compositional"][:2]
fig, ax = plt.subplots(len(sel), 3, figsize=(9, 3 * len(sel)))
for r, p in enumerate(sel):
    for c, m in enumerate(["base", "run1", "run2"]):
        ax[r, c].imshow(Image.open(f"work_p2/images/{m}/{p['id']}_s0.png")); ax[r, c].axis("off")
        ax[r, c].set_title(f"{m}: {p['prompt'][:34]}", fontsize=6)
plt.tight_layout(); plt.savefig("p2_grid.png", dpi=120); plt.show()'''),
 ("code", '''# upload to a Hugging Face dataset (prompt, seed, model in metadata)
from p2_finetune.hf_upload import push_images; push_images(META, "sivasuryachandran/sd15-naruto-lora-generations")'''),
 ("markdown", "## 2.4 Evaluation: KID + CLIP (primary), LPIPS diversity (secondary), efficiency"),
 ("code", '''from p2_finetune import evaluate as E
clip = E.Clip()
test_paths = [p for p, _ in test]; train_paths = [p for p, _ in data_prep.read_split("work_p2/data", "train")]
res = pd.DataFrame([E.summarize_model(clip, META, m, test_paths) for m in ["base", "run1", "run2"]]).set_index("model")
res["gpu_hours"] = [None, info1["gpu_hours"], info2["gpu_hours"]]
res["peak_gpu_mem_gb"] = [None, info1["peak_gpu_mem_gb"], info2["peak_gpu_mem_gb"]]
res'''),
 ("markdown", "KID is computed from 20 in-domain generations vs the test images, so it is noisy (see std). Fill in which run is final after looking at the table and the human evaluation."),
 ("code", '''FINAL = "run1"   # choose after reviewing results; justify in the report
from p2_finetune.human_eval import build_sheet
build_sheet(META, final=FINAL)    # share work_p2/human_eval/pair_*.png + rating_sheet_TEMPLATE.csv with 3 raters
# from p2_finetune.human_eval import win_rate; win_rate("work_p2/human_eval", ["rater1.csv", "rater2.csv", "rater3.csv"])'''),
 ("markdown", "### Failure modes (>= 2, with images)"),
 ("code", '''cand = E.failure_candidates(clip, META)
for m, rows in cand.items():
    print(m)
    for score, f, pr in rows: print("  ", round(score, 1), f, "|", pr[:70])
fig, ax = plt.subplots(1, 3, figsize=(10, 4))
for a, (m, rows) in zip(ax, cand.items()):
    a.imshow(Image.open(rows[0][1])); a.set_title(f"{m} (lowest CLIP): {rows[0][2][:40]}", fontsize=6); a.axis("off")
plt.savefig("p2_failures.png", dpi=120); plt.show()'''),
 ("markdown", "## C4 safety: license, NSFW filter, memorization"),
 ("code", '''print("images flagged by the SD safety checker:", res["nsfw_flagged"].to_dict())
top3, allpairs = E.memorization(clip, META, FINAL, train_paths)
fig, ax = plt.subplots(3, 2, figsize=(6, 9))
for r, p in enumerate(top3):
    ax[r, 0].imshow(Image.open(p["gen"])); ax[r, 1].imshow(Image.open(p["train"]))
    ax[r, 0].set_title(f"generated (cos={p['cosine']:.3f})", fontsize=7); ax[r, 1].set_title("closest training image", fontsize=7)
    ax[r, 0].axis("off"); ax[r, 1].axis("off")
plt.savefig("p2_memorization.png", dpi=120); plt.show()'''),
]

P3 = [
 ("markdown", "# Part 3 - Multimodal multi-agent system\nPlanner -> Vision -> Prompt Engineering -> Image Generation (SD1.5 img2img / InstructPix2Pix / Part 2 LoRA) -> Critique -> accept | revise (max 2)."),
 ("code", CELL1), ("code", SETUP),
 ("code", '''os.environ.setdefault("P3_LORA", "sivasuryachandran/sd15-naruto-lora-r4")   # Part 2 adapter on Hugging Face (or local path work_p2/runs/lora_r4)
from common.vlm import VLM
from p3_agents.models import Hub
from p3_agents.assets import make_assets
from p3_agents import graph, evaluate as E
hub = Hub(VLM())
make_assets()'''),
 ("markdown", "## Smoke test: one request through all agents"),
 ("code", '''s = graph.run(hub, "Make the sky purple.", "work_p3/assets/scene.png")
print(s.route, s.revisions, s.total_seconds, s.message)
json.dumps([{"agent": t["agent"], "seconds": t["seconds"]} for t in s.trace])'''),
 ("markdown", "## 3.3 Fifteen scenarios (defined in `p3_agents/scenarios.py` before running) - with and without the critic loop"),
 ("code", '''with_critic = E.run_scenarios(hub, enable_critic=True)
no_critic = E.run_scenarios(hub, enable_critic=False)
summary = pd.DataFrame({"with_critic": [with_critic.auto_pass.sum(), E.efficiency(with_critic)],
                        "without_critic": [no_critic.auto_pass.sum(), E.efficiency(no_critic)]}, index=["auto_pass_out_of_15", "median_sec_per_request"])
summary'''),
 ("code", '''with_critic[["id", "cat", "auto_pass", "route", "revisions", "critic_decision", "seconds", "message"]]'''),
 ("markdown", "Automatic checks are a first pass. Both members now fill `human_eval_p3.csv` (`pass_a`, `pass_b`, `agreed_pass` after resolving disagreements, `my_accept_or_revise` for critic agreement), then run `E.human_summary()`."),
 ("code", '''E.human_sheet(with_critic, no_critic)
# E.human_summary("human_eval_p3.csv")'''),
 ("markdown", "## 10 red-team cases"),
 ("code", '''rt = E.run_redteam(hub)
print("attack success:", int(rt.attack_success.fillna(False).sum()), "/ 10")
rt[["id", "kind", "attack_success", "refused", "route", "message"]]'''),
 ("markdown", "## Demo app (Gradio) - used in the team demo"),
 ("code", '''from p3_agents.app import build
demo = build(hub)
# demo.launch(share=True)   # uncomment to serve; optional bonus: deploy as a Hugging Face Space'''),
]

if __name__ == "__main__":
    os.makedirs("notebooks", exist_ok=True)
    for name, cells in [("p1_rag", P1), ("p2_finetune", P2), ("p3_agents", P3)]:
        json.dump(nb(cells), open(f"notebooks/{name}.ipynb", "w"), indent=1)
        print("wrote", name)
