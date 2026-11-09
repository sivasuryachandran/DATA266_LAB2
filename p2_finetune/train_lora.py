"""2.2 LoRA fine-tuning of Stable Diffusion 1.5 on one 16 GB GPU.

Both runs use identical data, steps, batch size, lr and seed; only the LoRA rank differs
(run 1: r=4, run 2: r=32). VAE latents and text embeddings are cached once (frees the VAE and
text encoder), the UNet trains with gradient checkpointing and fp16 autocast; LoRA weights are fp32.
"""
import json
import math
import os
import time

BASE = "stable-diffusion-v1-5/stable-diffusion-v1-5"
SEED = 266


def train(name, rank, data_dir, out_root="work_p2/runs", steps=1500, batch_size=4, lr=1e-4,
          warmup=50, base=BASE, seed=SEED, grad_clip=1.0):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict
    from PIL import Image
    from transformers import CLIPTextModel, CLIPTokenizer
    from p2_finetune.data_prep import read_split

    dev = "cuda"
    torch.manual_seed(seed)
    gen = torch.Generator(device=dev).manual_seed(seed)
    out = os.path.join(out_root, name)
    os.makedirs(out, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()

    tok = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer")
    text_enc = CLIPTextModel.from_pretrained(base, subfolder="text_encoder", torch_dtype=torch.float16).to(dev).eval()
    vae = AutoencoderKL.from_pretrained(base, subfolder="vae", torch_dtype=torch.float16).to(dev).eval()
    train_items = read_split(data_dir, "train")

    # cache latents (original + horizontal flip) and text embeddings once
    latents, embeds = [], []
    with torch.no_grad():
        for path, cap in train_items:
            img = Image.open(path).convert("RGB")
            x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 127.5 - 1
            x = torch.stack([x, x.flip(-1)]).to(dev, torch.float16)
            latents.append((vae.encode(x).latent_dist.sample(gen) * vae.config.scaling_factor).cpu())
            ids = tok(cap, padding="max_length", max_length=tok.model_max_length, truncation=True, return_tensors="pt").input_ids.to(dev)
            embeds.append(text_enc(ids)[0].cpu())
    latents = torch.stack(latents)           # N, 2, 4, 64, 64
    embeds = torch.cat(embeds)               # N, 77, 768
    del vae, text_enc
    torch.cuda.empty_cache()

    unet = UNet2DConditionModel.from_pretrained(base, subfolder="unet", torch_dtype=torch.float16).to(dev)
    unet.requires_grad_(False)
    unet.add_adapter(LoraConfig(r=rank, lora_alpha=rank, init_lora_weights="gaussian",
                                target_modules=["to_k", "to_q", "to_v", "to_out.0"]))
    params = [p for p in unet.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()
    n_trainable = sum(p.numel() for p in params)
    unet.enable_gradient_checkpointing()
    unet.train()
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    sched = DDPMScheduler.from_pretrained(base, subfolder="scheduler")
    scaler = torch.cuda.amp.GradScaler()
    lr_at = lambda s: lr * min(1, (s + 1) / warmup) * (0.5 * (1 + math.cos(math.pi * s / steps)) * 0.9 + 0.1)

    n = len(latents)
    losses, perm, ptr = [], torch.randperm(n, generator=torch.Generator().manual_seed(seed)), 0
    for step in range(steps):
        if ptr + batch_size > n:
            perm, ptr = torch.randperm(n), 0
        idx = perm[ptr:ptr + batch_size]
        ptr += batch_size
        flip = torch.randint(0, 2, (len(idx),))
        x0 = latents[idx, flip].to(dev)
        cond = embeds[idx].to(dev)
        noise = torch.randn(x0.shape, device=dev, generator=gen)
        t = torch.randint(0, sched.config.num_train_timesteps, (len(idx),), device=dev, generator=gen).long()
        xt = sched.add_noise(x0, noise, t)
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        with torch.autocast("cuda", dtype=torch.float16):
            pred = unet(xt, t, cond).sample
        loss = F.mse_loss(pred.float(), noise.float())
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, grad_clip)
        scaler.step(opt)
        scaler.update()
        losses.append(loss.item())
        if step % 100 == 0:
            print(f"[{name}] step {step}/{steps} loss {sum(losses[-100:]) / len(losses[-100:]):.4f}", flush=True)

    from diffusers import StableDiffusionPipeline
    StableDiffusionPipeline.save_lora_weights(out, unet_lora_layers=convert_state_dict_to_diffusers(get_peft_model_state_dict(unet)))
    info = {"run": name, "rank": rank, "lora_alpha": rank, "trainable_params": n_trainable, "learning_rate": lr,
            "steps": steps, "batch_size": batch_size, "train_images": n, "seed": seed,
            "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "train_seconds": round(time.time() - t_start, 1),
            "gpu_hours": round((time.time() - t_start) / 3600, 3), "final_loss_100": sum(losses[-100:]) / 100,
            "base_model": base}
    json.dump(info, open(os.path.join(out, "train_info.json"), "w"), indent=1)
    json.dump(losses, open(os.path.join(out, "losses.json"), "w"))
    del unet, opt
    torch.cuda.empty_cache()
    return info
