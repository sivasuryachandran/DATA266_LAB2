# AI coding assistant usage

- **Claude Code (Claude Sonnet 5.5)** was the only AI coding assistant used.
  - Initial scaffold: generated `common/`, `p1_rag/`, `p2_finetune/`, `p3_agents/`, `tools/build_notebooks.py` and `tests/` from the lab specification.
  - Part 1 development (commits `9bc21aa`, `99c10d6`, `98a2d5d`): table-row indexing, the 14B 4-bit answerer, the evidence check, the conflict pass, citation formatting, query decomposition and answer salvage.
  - Final-week support: ran the pipelines on a rented 24 GB cloud GPU, fixed first-run bugs (Chroma collection-name length, dependency pins), diagnosed Kaggle submissions against the PDFs, and wrote `tools/local_score.py`, a development-only checker with a hand-written key for the 40 public practice questions (it is not used by the pipeline and contains no answers in the answering code).
  - Experiments that were tried and **not** adopted are not part of the tagged code.
- Members: <edit: list what each member read, ran, wrote or changed by hand>.
- Hosted models used at run time: **none**. All models are open weights: Qwen/Qwen2.5-VL-7B-Instruct and Qwen2.5-VL-3B-Instruct (figure reading, Parts 1 and 3), Qwen2.5-14B-Instruct 4-bit (Part 1 answers), BAAI/bge-small-en-v1.5, Stable Diffusion 1.5, timbrooks/instruct-pix2pix, CLIP ViT-B/32.
- Claude Code also edited the team's shared report draft (Google Doc): it added Appendix A, filled the AI-usage and contributions entries for Siva, and replaced the surname placeholder.
