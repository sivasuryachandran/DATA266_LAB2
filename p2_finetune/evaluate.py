"""2.4 Metrics: KID + CLIP score (primary), LPIPS diversity (secondary), memorization check (C4)."""
import itertools

import numpy as np

CLIP_ID = "openai/clip-vit-base-patch32"


def _load(paths):
    from PIL import Image
    return [Image.open(p).convert("RGB") for p in paths]


def _uint8(imgs, size=299):
    import torch
    return torch.stack([torch.from_numpy(np.asarray(i.resize((size, size)))).permute(2, 0, 1) for i in imgs])


def kid(gen_paths, real_paths, subsets=50, seed=266):
    import torch
    from torchmetrics.image.kid import KernelInceptionDistance
    g = _load(gen_paths)
    sub = min(len(g), len(real_paths), 20)
    torch.manual_seed(seed)
    m = KernelInceptionDistance(subset_size=sub, subsets=subsets, normalize=False)
    m.update(_uint8(_load(real_paths)), real=True)
    m.update(_uint8(g), real=False)
    mean, std = m.compute()
    return float(mean), float(std)


class Clip:
    def __init__(self):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(CLIP_ID).to(self.dev).eval()
        self.proc = CLIPProcessor.from_pretrained(CLIP_ID)

    def image_emb(self, imgs, bs=32):
        import torch
        out = []
        for s in range(0, len(imgs), bs):
            x = self.proc(images=imgs[s:s + bs], return_tensors="pt").to(self.dev)
            with torch.no_grad():
                e = self.model.get_image_features(**x)
            out.append(torch.nn.functional.normalize(e, dim=-1).cpu())
        return torch.cat(out)

    def text_emb(self, texts):
        import torch
        x = self.proc(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77).to(self.dev)
        with torch.no_grad():
            e = self.model.get_text_features(**x)
        return torch.nn.functional.normalize(e, dim=-1).cpu()


def clip_scores(clip: Clip, meta_rows):
    """CLIP score = 100 * cosine(prompt, image) for each row."""
    ie = clip.image_emb(_load([r["file"] for r in meta_rows]))
    te = clip.text_emb([r["prompt"] for r in meta_rows])
    return (100 * (ie * te).sum(-1)).tolist()


def lpips_diversity(meta_rows):
    """Mean LPIPS between the two seeds of each prompt (higher = more diverse)."""
    import lpips
    import torch
    net = lpips.LPIPS(net="alex", verbose=False).cuda().eval()
    by = {}
    for r in meta_rows:
        by.setdefault(r["prompt_id"], {})[r["seed"]] = r["file"]
    to_t = lambda p: (torch.from_numpy(np.asarray(_load([p])[0])).permute(2, 0, 1).float() / 127.5 - 1)[None].cuda()
    d = []
    for v in by.values():
        a, b = (v[k] for k in sorted(v)[:2])
        with torch.no_grad():
            d.append(float(net(to_t(a), to_t(b))))
    return float(np.mean(d))


def summarize_model(clip, meta, model, real_test_paths):
    rows = [r for r in meta if r["model"] == model]
    ind = [r for r in rows if r["type"] == "in_domain"]
    km, ks = kid([r["file"] for r in ind], real_test_paths)
    cs = clip_scores(clip, rows)
    return {"model": model, "kid_mean": km, "kid_std": ks, "clip_score": float(np.mean(cs)),
            "clip_in_domain": float(np.mean([c for c, r in zip(cs, rows) if r["type"] == "in_domain"])),
            "clip_out_of_domain": float(np.mean([c for c, r in zip(cs, rows) if r["type"] == "out_of_domain"])),
            "clip_compositional": float(np.mean([c for c, r in zip(cs, rows) if r["type"] == "compositional"])),
            "lpips_diversity": lpips_diversity(rows), "nsfw_flagged": sum(r["nsfw_flagged"] for r in rows),
            "n_images": len(rows)}


def memorization(clip: Clip, meta, model, train_paths, k=3):
    """For the 10 in-domain images (seed 0) find the nearest training image by CLIP cosine; return the k closest pairs."""
    rows = [r for r in meta if r["model"] == model and r["type"] == "in_domain" and r["seed"] == 0][:10]
    ge = clip.image_emb(_load([r["file"] for r in rows]))
    te = clip.image_emb(_load(train_paths))
    sims = ge @ te.T
    best = sims.max(1)
    pairs = [{"gen": rows[i]["file"], "train": train_paths[int(sims[i].argmax())], "cosine": float(best.values[i]) if hasattr(best, "values") else float(best[i])}
             for i in range(len(rows))]
    return sorted(pairs, key=lambda p: -p["cosine"])[:k], pairs


def failure_candidates(clip, meta, models=("base", "run1", "run2")):
    """Lowest CLIP rows on out-of-domain / compositional prompts and the highest-LPIPS-collapse (lowest diversity) prompts."""
    out = {}
    for m in models:
        rows = [r for r in meta if r["model"] == m and r["type"] != "in_domain"]
        sc = clip_scores(clip, rows)
        out[m] = sorted(zip(sc, [r["file"] for r in rows], [r["prompt"] for r in rows]))[:3]
    return out
