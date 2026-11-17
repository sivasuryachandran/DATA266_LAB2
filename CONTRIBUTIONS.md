# Contributions - Pair_Programming_Team_06

Team members: Rajesh Paruchuri (GitHub `Paruchuri-Rajesh`) and Siva Surya Chandran (GitHub `sivasuryachandran`).
The team repository is https://github.com/Paruchuri-Rajesh/data266-lab2-team06; its `CONTRIBUTIONS.md` is the team-level file.
This file describes Siva's second, independent implementation, which lives in this repository (and in `sivasurya_chandran/` of the team repository).

| Part | Siva's work (this repository) |
|---|---|
| 1 RAG | Built a second Part 1 pipeline: hybrid dense + BM25 retrieval, table-first lookup, 4-bit Qwen2.5-VL-7B figure reader, 4-bit Qwen2.5-14B answerer with an evidence check and a conflict pass. Wrote and ran the 25-question golden-set evaluation (baseline 23, final 22 of 25), a groundedness rating page, and Kaggle submissions of this pipeline's output files (public scores 0.821, 0.794; the team repo records a later file, `submission_gh_v7`, at 0.853). |
| 2 Fine-tuning | Ran the complete Part 2 pipeline: rank-4 and rank-32 LoRA runs, 120 generated images, KID/CLIP/LPIPS, memorization check, failure modes; built the blind rating page used by three raters; wrote the final-run rationale and model-card text (`docs/`). |
| 3 Agents | Ran the complete Part 3 pipeline (15 scenarios with and without the critic, 10 red-team cases), fixed routing and a real-person-edit guardrail after the first run (first run and fixes kept in `notebooks/history/`), built the judging page. |
| Report | Report generator (`tools/build_report.py`, numbers read from notebook outputs) and Appendix A of the shared report draft. |

Both members ran a complete pipeline for every part (Rajesh's is in the team repository). An AI coding assistant helped with all of the above; see `AI_USAGE.md`.
Who led which part, who wrote which report sections, and the meeting recordings are recorded in the team repository's `CONTRIBUTIONS.md`.

## Meeting recordings
| Date | Topic | Link |
|---|---|---|
| <fill in: date> | <fill in: topic> | <fill in: link> |
