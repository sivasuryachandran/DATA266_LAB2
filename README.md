# DATA 266 Lab 2 - Pair_Programming_Team_06 (Siva Surya Chandran's implementation)
Part 1 multimodal RAG (Kaggle) · Part 2 LoRA fine-tuning of Stable Diffusion 1.5 · Part 3 multimodal multi-agent system.
Everything runs on one 16 GB GPU (Kaggle T4/P100 or Colab T4). Models are open weights only; **no hosted API is used**:
Part 1: unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit (reads figures at ingestion; falls back to Qwen/Qwen2.5-VL-3B-Instruct) and
unsloth/Qwen2.5-14B-Instruct-bnb-4bit (answers; pre-quantised 4-bit copy of Qwen/Qwen2.5-14B-Instruct, ~9.5 GB, loaded after the VLM is unloaded so both fit a T4);
Part 3: Qwen/Qwen2.5-VL-3B-Instruct; BAAI/bge-small-en-v1.5 (embeddings), stable-diffusion-v1-5/stable-diffusion-v1-5,
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
All three parts have been run on a GPU and their notebooks carry executed outputs.
* **Part 1** - `notebooks/p1_rag.ipynb` (final pipeline; Qwen2.5-VL-7B 4-bit reads figures, Qwen2.5-14B-Instruct 4-bit answers; both fit a 16 GB GPU). Kaggle public score of this pipeline: 0.821.
* **Part 2** - `notebooks/p2_finetune.ipynb` (RTX 5090 run). Blind human evaluation: `human_eval_p2/` (3 raters; `tools/make_rating_page.py` builds the rating page).
* **Part 3** - `notebooks/p3_agents.ipynb` (RTX 4090 run with a rank-4 adapter re-trained with the same code and settings as Part 2, since the Hugging Face copy is private). Earlier runs are kept in `notebooks/history/`. Traces and sample outputs: `artifacts/p3/`. Human judging: `human_eval_p3/` (`tools/make_p3_judge_page.py`).
* The Part 3 planner rules and guardrails were changed after a first run on the 15 scenarios and 10 red-team cases (10/15 and 2/10 attacks before; see `notebooks/history/`), so the final Part 3 numbers are optimistic and are reported with that caveat.
* Numbers in the report must be copied from executed notebook outputs, never typed by hand.

## Deliverables checklist
`golden_set.json` + `prompts.json` + `p3_agents/scenarios.py` committed before tuning/running · notebooks executed with outputs ·
tags `p1-v1.0` `p2-v1.0` `p3-v1.0` · HF models/dataset/model card · IEEE report (<=12 pages/part, diagrams, results tables, C4 safety, system card) ·
`CONTRIBUTIONS.md` · `AI_USAGE.md` · `links.txt` · zip `Pair_Programming_Team_##.zip`.

## Part 1 - changes since the first run (see git log)
* Models/env: `P1_VLM` (figure reader, default Qwen2.5-VL-7B 4-bit), `P1_LLM` (answerer, default Qwen2.5-14B-Instruct 4-bit, `none` = VLM answers), `LLM_QUANT` (`4bit`/`none`). Descriptions are cached per model in `work_p1/descriptions_<model>.json` (delete to re-describe).
* Tables are indexed whole and row by row (outturn vs forecast spelled out); table numbers are removed from plain-text chunks; retrieval is hybrid (dense + BM25, RRF) with one best hit per document; answers must quote verbatim evidence from the sources, otherwise status `not_found`; a second pass flags DOC1/DOC3 or DOC2/DOC3 disagreements as `conflict`.
* Citations name a figure/table by label and page (`DOC2:Figure A-3,DOC2:p4`), as in `sample_submission.csv`.
* The golden set was used while developing; for the report keep these numbers separate from any fresh held-out questions.

## Process notes and known limits (please read before grading)
* **Test sets and timing.** In this repository `golden_set.json` and `prompts.json` were first committed in commit `9bc21aa`, the same commit as Part 1 tuning changes,
  so this repository alone does not prove they were written before tuning. The team repository's own golden set (`part1/golden_set.csv`) was committed on Nov 15 before its tuning. `p3_agents/scenarios.py` was in the first commit.
* **Golden-set overlap.** Golden question G23 ("What was Japan's output growth in 2012?") uses the same wording as a public question about another country; it was not changed afterwards.
* **Tuned on the evaluation cases (Part 3).** Routing rules and the real-person-edit guardrail were added after a first run on the 15 scenarios and 10 red-team cases (10/15 and 2/10 attacks before; 15/15 and 0/10 after).
  Both executed notebooks are in `notebooks/history/`. The final numbers are optimistic.
* **Part 1 notebook vs tag.** `notebooks/p1_rag.ipynb` holds outputs from the code before the last Part 1 changes (golden set 22/25); `p1-v1.0` points at the later code. Later files from this pipeline were scored on Kaggle: 0.853 (`submission_gh_v7.csv`) and 0.858 (a later `submission.csv`, the team's best public score); the saved notebook was not re-executed on the final code because the first GPU instance ended.
* **Human evaluation.** Part 2: 3 raters, one of them outside the team (confirmed); win rate of the fine-tuned model 0.56 (style) and 0.44 (prompt match). Part 1 groundedness: 2 raters, means 3.80 and 3.48, within one point on 21 of 25 answers. Part 3 judging: 2 judges passed 10 and 7 of 15 scenarios (automatic checks: 15), agreed on 10 of 15 verdicts, disagreed on S02, S05, S07, S08 and S12, and both passed 6 of 15. Neither judge wrote a reason for the disagreements, so no discussion outcome is claimed: they are settled by a fixed conservative rule (a scenario counts as passed only if both judges passed it; S01, S03, S04, S09, S10, S11 pass, the five disputed ones do not; a lenient either-judge rule would give 11 of 15).
* **Not in the repository.** The hand-written answer key for the public practice questions used for development scoring is kept out on purpose. No hosted model API is used.
* **Gradio demo.** `p3_agents.app.build(hub).launch(share=True)` (see `notebooks/p3_agents.ipynb`). Install `pydantic==2.10.6` with the pinned Gradio: with a newer pydantic the page fails to launch.
  Tested on a Kaggle T4: three requests (style change, domain-style drawing with the Part 2 adapter, an unsafe request that is refused) ran correctly; the first request takes about five minutes because the image models load on first use, so warm the app up before a live demo.
