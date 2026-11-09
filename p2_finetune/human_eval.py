"""2.4 Blinded human evaluation: 25 randomized (final vs base) pairs, >=3 raters."""
import json
import os
import random

import pandas as pd


def build_sheet(meta, final="run1", out_dir="work_p2/human_eval", n=25, seed=266):
    from PIL import Image
    rng = random.Random(seed)
    base = {(r["prompt_id"], r["seed"]): r for r in meta if r["model"] == "base"}
    fin = {(r["prompt_id"], r["seed"]): r for r in meta if r["model"] == final}
    keys = rng.sample(sorted(base), n)
    os.makedirs(out_dir, exist_ok=True)
    key_rows, sheet = [], []
    for i, k in enumerate(keys):
        left_is_final = rng.random() < 0.5
        a, b = (fin[k], base[k]) if left_is_final else (base[k], fin[k])
        ia, ib = Image.open(a["file"]), Image.open(b["file"])
        canvas = Image.new("RGB", (ia.width + ib.width + 10, ia.height), "white")
        canvas.paste(ia, (0, 0))
        canvas.paste(ib, (ia.width + 10, 0))
        canvas.save(os.path.join(out_dir, f"pair_{i:02d}.png"))
        sheet.append({"pair": i, "prompt": base[k]["prompt"], "better_domain_style(L/R)": "", "better_prompt_match(L/R)": ""})
        key_rows.append({"pair": i, "left": "final" if left_is_final else "base"})
    pd.DataFrame(sheet).to_csv(os.path.join(out_dir, "rating_sheet_TEMPLATE.csv"), index=False)
    pd.DataFrame(key_rows).to_csv(os.path.join(out_dir, "KEY_do_not_share.csv"), index=False)


def win_rate(out_dir, rater_files):
    """rater_files: CSVs (one per rater) copied from the template and filled with L or R."""
    key = pd.read_csv(os.path.join(out_dir, "KEY_do_not_share.csv")).set_index("pair")["left"]
    res = {}
    for col in ("better_domain_style(L/R)", "better_prompt_match(L/R)"):
        wins, tot = 0, 0
        for f in rater_files:
            df = pd.read_csv(f)
            for _, r in df.iterrows():
                v = str(r[col]).strip().upper()
                if v in ("L", "R"):
                    tot += 1
                    wins += (v == "L") == (key[r["pair"]] == "final")
        res[col] = {"final_win_rate": wins / max(tot, 1), "votes": tot}
    return res
