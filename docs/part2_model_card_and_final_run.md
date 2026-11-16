# Part 2 - final run choice and model card text

All numbers below are copied from the executed output of `notebooks/p2_finetune.ipynb` (cells 7 and 14).

## Results (20 prompts x 2 seeds = 40 images per model, same sampler: DPM-Solver++, 30 steps, guidance 7.5, 512x512)

| model | KID (mean +- std) | CLIP score | CLIP in-domain | CLIP out-of-domain | CLIP compositional | LPIPS diversity | safety-checker flags |
|---|---|---|---|---|---|---|---|
| base SD 1.5 | 0.1230 +- 0.0141 | 33.35 | 32.19 | 33.52 | 35.49 | 0.8215 | 0 |
| run 1: LoRA r=4 | 0.0455 +- 0.0088 | 31.76 | 29.70 | 32.07 | 35.58 | 0.7943 | 0 |
| run 2: LoRA r=32 | 0.0259 +- 0.0074 | 30.88 | 28.87 | 32.62 | 33.18 | 0.8005 | 3 |

Training (same data, 1500 steps, batch 4, lr 1e-4, seed 266):
r=4 has 797,184 trainable parameters, 2.66 GB peak memory, 519 s (0.144 GPU-h);
r=32 has 6,377,472 trainable parameters, 2.73 GB peak memory, 728 s (0.202 GPU-h).

## Final run: run 1 (LoRA rank 4) - proposed, to be confirmed by the team
- Both LoRA runs move generations far closer to the training domain than the base model (KID 0.123 -> 0.045 / 0.026). Run 2 has the lower KID (about two standard deviations below run 1; KID is noisy with 20 in-domain images).
- Run 1 follows prompts better overall (CLIP 31.8 vs 30.9; in-domain 29.7 vs 28.9; compositional 35.6 vs 33.2). On out-of-domain prompts run 2 scores slightly higher (32.6 vs 32.1).
- Run 1 produced no safety-checker flags (run 2: 3 of 40 images were blanked).
- Run 1 is cheaper (0.144 vs 0.202 GPU-hours) and an 8x smaller adapter.
- Fine-tuned CLIP scores are below the base model's because CLIP scores a stylised cartoon lower against a photographic caption; they should be read with KID and the human evaluation.
- Human evaluation (run 1 vs base, blind, 25 pairs, 3 raters; files in `human_eval_p2/`): pooled win rate of the fine-tuned model 0.56 for domain style (42/75) and 0.44 for prompt match (33/75); per rater 14, 15, 13 of 25 (style) and 12, 11, 10 of 25 (prompt match). Pairwise agreement 0.68 (style) and 0.57 (prompt match). The images come from the Vast re-run of Part 2 (base and rank-4 adapter). Confirm that one rater is outside the team.

## Model card text for Hugging Face (push with `push_adapter(..., results_text=<the table above>)`)
**sd15-naruto-lora-r4** - LoRA adapter (rank 4) for Stable Diffusion 1.5 (`stable-diffusion-v1-5/stable-diffusion-v1-5`), fine-tuned on `lambdalabs/naruto-blip-captions` (1,221 images; 1,099 train / 122 test, seed 266; no corrupted or duplicate images found).
Training: 1,500 steps, batch size 4, lr 1e-4, 797,184 trainable parameters, 2.66 GB peak GPU memory, 519 s.
Results: KID 0.0455 (base 0.1230), CLIP score 31.76 (base 33.35), LPIPS diversity 0.794 (base 0.822), 0/40 images flagged by the safety checker.
Limitations: biased to one franchise and mostly front-facing characters; weaker on out-of-domain and multi-character compositional prompts; outputs can resemble training images; the Stable Diffusion safety checker stays on but is not a guarantee.
License: **non-commercial research use only** (copyrighted characters; dataset license CC BY-NC-SA 4.0; inherits the CreativeML OpenRAIL-M terms of the base model).
