"""2.3 Generate 2 images (seeds 0,1) per prompt for base / run 1 / run 2 with identical sampler settings."""
import json
import os

BASE = "stable-diffusion-v1-5/stable-diffusion-v1-5"
SETTINGS = {"steps": 30, "guidance_scale": 7.5, "height": 512, "width": 512, "scheduler": "DPMSolverMultistep"}


def load_pipe(base=BASE):
    import torch
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(base, torch_dtype=torch.float16).to("cuda")   # safety checker stays ON
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def generate_all(models: dict, prompts, out_dir="work_p2/images", seeds=(0, 1), base=BASE):
    """models: {'base': None, 'run1': path_or_repo, 'run2': path_or_repo}. Returns metadata list.
    Images flagged by the safety checker come back black; they are recorded in 'nsfw_flagged'."""
    import torch
    pipe = load_pipe(base)
    meta = []
    for name, lora in models.items():
        os.makedirs(os.path.join(out_dir, name), exist_ok=True)
        if lora:
            pipe.load_lora_weights(lora)
        for p in prompts:
            for s in seeds:
                g = torch.Generator("cuda").manual_seed(s)
                r = pipe(p["prompt"], generator=g, num_inference_steps=SETTINGS["steps"],
                         guidance_scale=SETTINGS["guidance_scale"], height=512, width=512)
                path = os.path.join(out_dir, name, f"{p['id']}_s{s}.png")
                r.images[0].save(path)
                meta.append({"model": name, "prompt_id": p["id"], "type": p["type"], "prompt": p["prompt"],
                             "seed": s, "file": path, "nsfw_flagged": bool(r.nsfw_content_detected and r.nsfw_content_detected[0])})
        if lora:
            pipe.unload_lora_weights()
    json.dump(meta, open(os.path.join(out_dir, "metadata.json"), "w"), indent=1)
    return meta
