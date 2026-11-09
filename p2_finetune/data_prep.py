"""2.1 Load captioned dataset, drop corrupted + duplicate images, split 90/10 (seed 266), resize 512."""
import hashlib
import io
import json
import os

import numpy as np

SEED = 266
DATASET = "lambdalabs/naruto-blip-captions"   # CC BY-NC-SA 4.0 per the dataset card -- verify before use
RES = 512


def dhash(img, size=8):
    g = np.asarray(img.convert("L").resize((size + 1, size)), dtype=np.int16)
    return int("".join("1" if b else "0" for b in (g[:, 1:] > g[:, :-1]).flatten()), 2)


def hamming(a, b):
    return bin(a ^ b).count("1")


def clean_and_split(dataset_name=DATASET, out_dir="work_p2/data", dup_hamming=2):
    from datasets import load_dataset
    from PIL import Image
    ds = load_dataset(dataset_name, split="train")
    os.makedirs(os.path.join(out_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "test"), exist_ok=True)
    kept, md5s, hashes = [], set(), []
    stats = {"total": len(ds), "corrupted": 0, "exact_duplicates": 0, "near_duplicates": 0}
    for i, row in enumerate(ds):
        try:
            img = row["image"]
            img.load()
            img = img.convert("RGB")
        except Exception:
            stats["corrupted"] += 1
            continue
        h5 = hashlib.md5(img.tobytes()).hexdigest()
        if h5 in md5s:
            stats["exact_duplicates"] += 1
            continue
        dh = dhash(img)
        if any(hamming(dh, o) <= dup_hamming for o in hashes):
            stats["near_duplicates"] += 1
            continue
        md5s.add(h5)
        hashes.append(dh)
        kept.append((i, img, row["text"]))
    stats["removed_total"] = stats["total"] - len(kept)
    stats["kept"] = len(kept)
    rng = np.random.RandomState(SEED)
    order = rng.permutation(len(kept))
    n_test = max(1, round(0.1 * len(kept)))
    splits = {"test": order[:n_test], "train": order[n_test:]}
    for name, idxs in splits.items():
        meta = []
        for j in idxs:
            i, img, cap = kept[j]
            w, h = img.size
            s = min(w, h)
            img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((RES, RES), Image.LANCZOS)
            fn = f"{i:05d}.png"
            img.save(os.path.join(out_dir, name, fn))
            meta.append({"file_name": fn, "text": cap})
        with open(os.path.join(out_dir, name, "metadata.jsonl"), "w") as f:
            f.write("\n".join(json.dumps(m) for m in meta))
        stats[name] = len(meta)
    json.dump(stats, open(os.path.join(out_dir, "stats.json"), "w"), indent=1)
    return stats


def read_split(out_dir, name):
    from PIL import Image
    d = os.path.join(out_dir, name)
    meta = [json.loads(l) for l in open(os.path.join(d, "metadata.jsonl")) if l.strip()]
    return [(os.path.join(d, m["file_name"]), m["text"]) for m in meta]
