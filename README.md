# DATA 266 Lab 2 - Pair_Programming_Team_##
Part 1 multimodal RAG (Kaggle) · Part 2 LoRA fine-tuning of Stable Diffusion 1.5 · Part 3 multimodal multi-agent system.
Everything runs on one 16 GB GPU (Kaggle T4/P100 or Colab T4). Models are open weights only; **no hosted API is used**:
Qwen/Qwen2.5-VL-3B-Instruct (VLM/LLM, Parts 1 and 3), BAAI/bge-small-en-v1.5 (embeddings), stable-diffusion-v1-5/stable-diffusion-v1-5,
timbrooks/instruct-pix2pix, openai/clip-vit-base-patch32, lpips (alex). Optional bonus reranker: BAAI/bge-reranker-base.

## Layout
```
common/        run-info cell, seeding, shared Qwen2.5-VL wrapper
p1_rag/        ingest.py (PyMuPDF, figure crops, VLM descriptions) · index.py (Chroma + BM25 + RRF) · rag.py (answers, conversations)
               golden.py (25-question golden set tooling + metrics) · run_competition.py (writes submission.csv)
p2_finetune/   data_prep.py · prompts.py · train_lora.py · generate.py · evaluate.py · human_eval.py · hf_upload.py
p3_agents/     agents.py (5 agents) · graph.py (orchestration, traces) · safety.py · app.py (Gradio) · scenarios.py (15 + 10) · evaluate.py
notebooks/     p1_rag.ipynb  p2_finetune.ipynb  p3_agents.ipynb   (built by tools/build_notebooks.py)
tests/         python -m unittest tests.test_logic   (CPU-only logic tests: label parsing, fusion, sanitising, guardrails)
```

## Run
```bash
pip install -r requirements.txt
python -m unittest tests.test_logic          # sanity check, no GPU needed
```
* **Part 1** - open `notebooks/p1_rag.ipynb` on Kaggle with the competition data attached (or `DATA_DIR=...`), Restart & Run All.
  It ingests, indexes, evaluates on `golden_set.json` (baseline vs final) and writes `submission.csv`. Headless: `python -m p1_rag.run_competition`.
  You must write `golden_set.json` yourselves from the PDFs (schema in `p1_rag/golden.py`) and commit it before tuning.
* **Part 2** - `notebooks/p2_finetune.ipynb` (about 2 x 35-50 min on a T4 at 1500 steps; adjust `steps`). Set `HF_TOKEN` to push adapters/images.
* **Part 3** - `notebooks/p3_agents.ipynb`; set `P3_LORA` to your Part 2 adapter. `p3_agents.app.build(hub).launch(share=True)` for the demo.

## Design choices (short)
* **P1**: dense (bge-small) + BM25 fused with RRF; figures retrieved by exact label first (`Figure B-5` and `Figure B-5 (corrected)` are separate objects; a corrected sibling is attached and flagged so the revision rule can fire); Appendix C pages are tagged `untrusted` at ingestion and filtered out of both indexes; retrieved text and VLM figure transcripts are sanitised for instruction-like sentences; answers are validated so citations/figure labels must come from retrieved sources. Conversation memory lives in a per-`conversation_id` `Chat` object; the vector store is never written during answering (asserted in the notebook). No question-specific rules: the status rules are in the prompt and generic regex cues.
* **P2**: custom diffusers+PEFT loop (cached VAE latents and text embeddings, gradient checkpointing, fp16 autocast, fp32 LoRA weights). Runs differ only in LoRA rank (4 vs 32). SD safety checker stays on.
* **P3**: plain-Python orchestration over one typed `State`; one diffusion pipeline resident at a time (VLM stays loaded); JSON trace per request in `work_p3/traces/`.

## Status of this repository
Written without access to the Part 1 data files or a GPU: the logic tests pass, but the model-dependent code
(ingestion on the real PDFs, training, generation, agents) has **not been executed yet**. Expect to tune figure
cropping in `ingest.py`, the CLIP accept threshold in `p3_agents/agents.py`, and pins in `requirements.txt` on the first GPU run.
No number in the report may be typed by hand: copy them from executed notebook outputs.

## Deliverables checklist
`golden_set.json` + `prompts.json` + `p3_agents/scenarios.py` committed before tuning/running · notebooks executed with outputs ·
tags `p1-v1.0` `p2-v1.0` `p3-v1.0` · HF models/dataset/model card · IEEE report (<=12 pages/part, diagrams, results tables, C4 safety, system card) ·
`CONTRIBUTIONS.md` · `AI_USAGE.md` · `links.txt` · zip `Pair_Programming_Team_##.zip`.
