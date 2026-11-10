"""Push adapters, model card and the image dataset to Hugging Face (token from env HF_TOKEN; never committed)."""
import json
import os

MODEL_CARD = """---
license: other
license_name: non-commercial-research-only
base_model: stable-diffusion-v1-5/stable-diffusion-v1-5
tags: [text-to-image, lora, diffusers, stable-diffusion]
datasets: [{dataset}]
---
# {name}
LoRA adapter (rank {rank}) for Stable Diffusion 1.5 fine-tuned on captioned images of {dataset}.
**Non-commercial research use only.** The characters are copyrighted; the dataset license is {dataset_license}.

## Training
steps {steps}, batch size {batch_size}, lr {learning_rate}, trainable params {trainable_params:,}, seed {seed},
peak GPU memory {peak_gpu_mem_gb} GB, {train_seconds} s on one 16 GB GPU.

## Results (see report for the full table)
{results}

## Limitations
Style is biased towards the training set (mostly front-facing anime characters); out-of-domain and
multi-character compositional prompts degrade; outputs can resemble training images (memorization check in report);
the SD safety checker is kept on but is not a guarantee. Inherits the CreativeML OpenRAIL-M restrictions of the base model.
"""


def push_adapter(run_dir, repo_id, results_text="", dataset="lambdalabs/naruto-blip-captions", dataset_license="CC BY-NC-SA 4.0"):
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id, exist_ok=True, private=True)
    info = json.load(open(os.path.join(run_dir, "train_info.json")))
    info.update(dataset=dataset, dataset_license=dataset_license, results=results_text, name=repo_id.split("/")[-1])
    open(os.path.join(run_dir, "README.md"), "w").write(MODEL_CARD.format(**info))
    api.upload_folder(folder_path=run_dir, repo_id=repo_id, ignore_patterns=["losses.json"])


def push_images(meta, repo_id):
    from datasets import Dataset, Image
    ds = Dataset.from_list([{"image": m["file"], "prompt": m["prompt"], "seed": m["seed"], "model": m["model"],
                             "prompt_type": m["type"]} for m in meta]).cast_column("image", Image())
    ds.push_to_hub(repo_id, token=os.environ["HF_TOKEN"], private=True)
