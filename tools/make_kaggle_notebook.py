"""Builds kaggle_final/p1_final.ipynb: a self-contained Kaggle notebook that runs the Part 1 pipeline end to end.

The pipeline source (common/ and p1_rag/) is embedded as readable %%writefile cells, so the notebook needs no GitHub access.
It finds the competition data under /kaggle/input, reads QUESTIONS_PATH (default: Questions.csv next to the manifest) and writes submission.csv.
"""
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
COMMIT = sys.argv[1] if len(sys.argv) > 1 else "2698425"
FILES = ["common/__init__.py", "common/run_info.py", "common/vlm.py", "p1_rag/__init__.py", "p1_rag/text_utils.py",
         "p1_rag/ingest.py", "p1_rag/index.py", "p1_rag/rag.py", "p1_rag/pipeline.py", "p1_rag/run_competition.py"]


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src.splitlines(True)}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(True)}


cells = [md("# DATA 266 Lab 2, Part 1 - multimodal RAG (Pair_Programming_Team_06, final notebook)\n"
            "Attach the competition data (done by the competition) and turn Internet on. Run All: the notebook installs pinned libraries, writes the "
            "pipeline code (cells below), reads `QUESTIONS_PATH` (default `Questions.csv` beside `kb_manifest.json`) and writes `/kaggle/working/submission.csv` with no manual step. "
            "Models: Qwen2.5-VL-7B-Instruct (4-bit, figure reader), Qwen2.5-14B-Instruct (4-bit, answers), bge-small-en-v1.5 (embeddings); no hosted API.\n")]
cells.append(code("# Cell 1 - run information (required)\n"
                  "import sys, random, numpy as np, torch\n"
                  "SEED = 266; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)\n"
                  f"CODE = 'github.com/sivasuryachandran/DATA266_LAB2 commit {COMMIT} (tag p1-v1.0); the Part 1 code is embedded in the cells below'\n"
                  "print({'code': CODE, 'python': sys.version.split()[0], 'torch': torch.__version__,\n"
                  "       'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu', 'seed': SEED})\n"))
cells.append(code("!pip install -q pymupdf rank_bm25 'chromadb==1.5.9' sentence-transformers==3.4.1 transformers==4.49.0 accelerate==1.3.0 bitsandbytes pandas 2>&1 | tail -3\n"))
cells.append(code("import os\nfor d in ('common', 'p1_rag'):\n    os.makedirs(f'/kaggle/working/lab2/{d}', exist_ok=True)\n"))
for f in FILES:
    src = open(os.path.join(ROOT, f), encoding="utf-8").read()
    cells.append(code(f"%%writefile /kaggle/working/lab2/{f}\n{src}"))
cells.append(code("import os, sys, time, glob\n"
                  "os.environ['HF_HOME'] = '/tmp/hf'\n"
                  "sys.path.insert(0, '/kaggle/working/lab2'); os.chdir('/kaggle/working')\n"
                  "from p1_rag.ingest import find_data_dir\n"
                  "from p1_rag.pipeline import build, answer_file\n"
                  "data_dir = find_data_dir()\n"
                  "QUESTIONS_PATH = os.environ.get('QUESTIONS_PATH', os.path.join(data_dir, 'Questions.csv'))   # graders swap in the hidden set\n"
                  "print('data:', data_dir, '| questions:', QUESTIONS_PATH)\n"
                  "t0 = time.time(); system, items, figures = build(data_dir=data_dir, work_dir='/kaggle/working/work_p1')\n"
                  "print('index built in', round((time.time() - t0) / 60, 1), 'min;', len(items), 'items,', len(figures), 'figures/tables')\n"
                  "t1 = time.time(); out, timing = answer_file(system, QUESTIONS_PATH, '/kaggle/working/submission.csv', with_time=True)\n"
                  "print('answered', len(out), 'questions in', round((time.time() - t1) / 60, 1), 'min; total', round((time.time() - t0) / 60, 1), 'min')\n"))
cells.append(code("import pandas as pd, subprocess\n"
                  "sub = pd.read_csv('/kaggle/working/submission.csv', dtype=str, keep_default_na=False)\n"
                  "print(sub.shape, sub['status'].value_counts().to_dict())\n"
                  "v = os.path.join(data_dir, 'validate_submission.py')\n"
                  "if os.path.exists(v):\n"
                  "    r = subprocess.run([sys.executable, v, '/kaggle/working/submission.csv', QUESTIONS_PATH], text=True, capture_output=True); print((r.stdout + r.stderr).strip())\n"
                  "sub.head()\n"))
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
os.makedirs(os.path.join(ROOT, "kaggle_final"), exist_ok=True)
json.dump(nb, open(os.path.join(ROOT, "kaggle_final", "p1_final.ipynb"), "w"), indent=1)
json.dump({"id": "sivasuryachandran/data266-team06-p1-final", "title": "data266-team06-p1-final", "code_file": "p1_final.ipynb", "language": "python",
           "kernel_type": "notebook", "is_private": False, "enable_gpu": True, "enable_tpu": False, "enable_internet": True, "dataset_sources": [],
           "competition_sources": ["data-266-fall-2026-lab-2-part-1-multimodal-rag"], "kernel_sources": [], "model_sources": []},
          open(os.path.join(ROOT, "kaggle_final", "kernel-metadata.json"), "w"), indent=2)
print("wrote kaggle_final/p1_final.ipynb with", len(cells), "cells,", os.path.getsize(os.path.join(ROOT, "kaggle_final", "p1_final.ipynb")), "bytes")
