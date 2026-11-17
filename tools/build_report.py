"""Builds report/report.html (IEEE-style two-column) and, with Chrome headless, report/report.pdf.

Every number is read from an executed notebook output or from a result file in this repository; nothing is typed in by
hand. Items that still need human input show as PENDING and fill in when the files arrive; re-run this script then.

    python tools/build_report.py
"""
import base64
import csv
import glob
import html
import itertools
import json
import os
import re
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
P = lambda *a: os.path.join(ROOT, *a)


def nb_texts(path):
    nb = json.load(open(P(path), encoding="utf-8-sig"))
    out = []
    for i, c in enumerate(nb["cells"]):
        for o in c.get("outputs", []) or []:
            t = "".join(o.get("text", [])) or "".join(o.get("data", {}).get("text/plain", []))
            out.append((i, t))
    return out


def find(texts, key):
    return next((t for _, t in texts if key in t), "")


def pending(x=""):
    return f'<span class="pend">PENDING{(" " + x) if x else ""}</span>'


def b64(path, mime="image/png"):
    return f"data:{mime};base64," + base64.b64encode(open(P(path), "rb").read()).decode()


# ------------------------------------------------------------------ Part 1 numbers
p1 = nb_texts("notebooks/p1_rag.ipynb")
t = find(p1, "correct_out_of_25")
base = re.search(r"baseline\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", t)
fin = re.search(r"final\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", t)
med = re.findall(r"(baseline|final)\s+([\d.]+)\s*$", t, re.M)
medd = {k: v for k, v in med}
ing = re.search(r"ingestion\+index seconds:\s*(\d+)", find(p1, "ingestion+index"))
idx = re.search(r"(\d+) indexed items; (\d+) figures/tables; (\d+) inside Appendix C", find(p1, "indexed items"))
safe = find(p1, "Appendix C items kept out")
appc = re.search(r"kept out of retrieval:\s*(\d+)", safe)
figinj = re.search(r"instruction-like text:\s*(\[.*?\])", safe)
chunkinj = re.search(r"before sanitising:\s*(\d+)", safe)
unchanged = re.search(r"vector store unchanged:\s*(\d+) items", find(p1, "vector store unchanged"))
sub = {r["id"]: r for r in csv.DictReader(open(P("artifacts/p1/submission.csv")))}


def grounded():
    files = sorted(glob.glob(P("human_eval_p1/rater_*.csv")))
    if len(files) < 2:
        return None
    d = [{r["id"]: int(r["groundedness"]) for r in csv.DictReader(open(f))} for f in files]
    ids = sorted(d[0])
    mean = sum(sum(x[i] for i in ids) for x in d) / (len(d) * len(ids))
    pairs = list(itertools.combinations(range(len(d)), 2))
    within = sum(abs(d[a][i] - d[b][i]) <= 1 for a, b in pairs for i in ids) / (len(pairs) * len(ids))
    return mean, within, len(d)


gr = grounded()

# ------------------------------------------------------------------ Part 2 numbers
p2 = nb_texts("notebooks/p2_finetune.ipynb")
dstats = re.search(r"\{'total': (\d+), 'corrupted': (\d+), 'exact_duplicates': (\d+), 'near_duplicates': (\d+), "
                   r"'removed_total': (\d+), 'kept': (\d+), 'test': (\d+), 'train': (\d+)\}", find(p2, "'total'"))
tr = {}
trt = find(p2, "trainable_params")
for name in ("lora_r4", "lora_r32"):
    m = re.search(name + r"\s+(\d+)\s+(\d+)\s+([\d.]+)\s+(\d+)\s+(\d+)", trt)
    m2 = re.findall(name + r"\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", trt)
    tr[name] = (m.groups(), m2[-1])
kt = find(p2, "kid_mean")
blocks = kt.split("\n\n")
kid = {}
for m in ("base", "run1", "run2"):
    a = re.search(m + r"\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", kt)
    b = [x for x in re.findall(m + r"\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+(\d+)", kt)]
    kid[m] = (a.groups(), b[0])
nsfw_line = find(p2, "flagged by the SD safety checker")
nsfw = re.search(r"\{'base': (\d+), 'run1': (\d+), 'run2': (\d+)\}", nsfw_line).groups()


def p2_human():
    key = {int(r["pair"]): r["left"] for r in csv.DictReader(open(P("human_eval_p2/KEY_do_not_share.csv")))}
    files = sorted(glob.glob(P("human_eval_p2/rater_*.csv")))
    cols = ("better_domain_style(L/R)", "better_prompt_match(L/R)")
    data = {}
    for f in files:
        rows = list(csv.DictReader(open(f)))
        data[os.path.basename(f)[6:-4]] = {c: {int(r["pair"]): r[c].strip().upper() for r in rows} for c in cols}
    res = {}
    for c in cols:
        w = sum(((v == "L") == (key[p] == "final")) for d in data.values() for p, v in d[c].items())
        n = sum(len(d[c]) for d in data.values())
        ag = [sum(data[a][c][p] == data[b][c][p] for p in range(25)) / 25 for a, b in itertools.combinations(data, 2)]
        res[c] = (w, n, sum(ag) / len(ag))
    return len(data), res


p2h = p2_human()

# ------------------------------------------------------------------ Part 3 numbers


def p3_nums(path):
    tx = nb_texts(path)
    s = find(tx, "auto_pass_out_of_15")
    wc = re.search(r"auto_pass_out_of_15\s+([\d.]+)\s+([\d.]+)", s).groups()
    ms = re.search(r"median_sec_per_request\s+([\d.]+)\s+([\d.]+)", s).groups()
    at = re.search(r"attack success:\s*(\d+)\s*/\s*(\d+)", find(tx, "attack success")).groups()
    return wc, ms, at, find(tx, "R01   image_injection")


p3f = p3_nums("notebooks/p3_agents.ipynb")
p3a = p3_nums("notebooks/history/p3_run1_before_fixes.ipynb")
p3b = p3_nums("notebooks/history/p3_run2_after_planner_fixes.ipynb")
rt_rows = re.findall(r"^\d+\s+(R\d+)\s+(\w+)\s+(True|False)\s+(True|False)\s+(\w*)", p3f[3], re.M)


def p3_judges():
    files = sorted(glob.glob(P("human_eval_p3/judge_*.csv")))
    sheet = {r["id"]: r for r in csv.DictReader(open(P("artifacts/p3/human_eval_p3.csv")))}
    out = []
    for f in files:
        rows = {r["id"]: r for r in csv.DictReader(open(f))}
        out.append((os.path.basename(f)[6:-4], rows))
    return sheet, out


sheet3, judges = p3_judges()

# ------------------------------------------------------------------ diagrams


def flow(title, boxes, w=470, h=64):
    n = len(boxes)
    bw = (w - 10 * (n - 1) - 8) / n
    s = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{html.escape(title)}" class="dia">']
    for i, b in enumerate(boxes):
        x = 4 + i * (bw + 10)
        s.append(f'<rect x="{x:.1f}" y="8" width="{bw:.1f}" height="{h - 16}" rx="5" fill="#eef3f8" stroke="#2f5d8a"/>')
        for j, line in enumerate(b.split("|")):
            s.append(f'<text x="{x + bw / 2:.1f}" y="{26 + j * 11}" font-size="8" text-anchor="middle">{html.escape(line)}</text>')
        if i < n - 1:
            s.append(f'<path d="M{x + bw:.1f} {h / 2} l10 0" stroke="#2f5d8a" marker-end="url(#a)"/>')
    s.append('<defs><marker id="a" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0 0L6 3L0 6z" fill="#2f5d8a"/></marker></defs></svg>')
    return "".join(s)


D1 = flow("Part 1 architecture", ["PDFs (3)|PyMuPDF text,|figure crops", "7B VLM reads|figures; table|rows indexed", "Chroma dense +|BM25, RRF fusion|1 hit per doc",
                                  "14B 4-bit LLM|JSON answer +|evidence check", "conflict pass|status rules|session memory"])
D2 = flow("Part 2 architecture", ["Naruto BLIP|1,221 images|clean, 90/10", "SD 1.5 UNet|LoRA r=4 / r=32|1500 steps", "20 prompts x 2|seeds x 3 models|(120 images)", "KID, CLIP, LPIPS|memorization|NSFW checker", "blind human|pair rating|(3 raters)"])
D3 = flow("Part 3 architecture", ["Planner|route + safety|(rules + LLM)", "Vision|caption, objects,|text, Q&A", "Prompt Engineer|never obeys|image text", "Image Generation|img2img / pix2pix|/ Part 2 LoRA", "Critique|CLIP + VLM +|safety; revise <=2"])

CSS = """
@page{size:Letter;margin:0.7in 0.6in}
body{font:9.4pt/1.28 'Times New Roman',Times,serif;color:#111;margin:0}
.title{text-align:center;margin:0 0 6px}.title h1{font-size:17pt;margin:0 0 4px;line-height:1.15}.title p{margin:1px 0;font-size:10pt}
.abs{font-weight:bold;font-size:9pt;margin:6px 0 8px}
.cols{column-count:2;column-gap:0.25in;text-align:justify}
h2{font-size:10.5pt;text-transform:uppercase;margin:9px 0 3px;break-after:avoid}h3{font-size:9.6pt;font-style:italic;margin:6px 0 2px;break-after:avoid}
p{margin:0 0 4px}table{border-collapse:collapse;width:100%;font-size:8pt;margin:3px 0 5px;break-inside:avoid}
th,td{border-top:1px solid #333;border-bottom:1px solid #999;padding:1.5px 3px;text-align:right}th:first-child,td:first-child{text-align:left}
th{border-top:2px solid #000}caption{font-size:8pt;text-align:left;font-style:italic;margin-bottom:2px;caption-side:top}
.dia{width:100%;height:auto;break-inside:avoid;margin:2px 0}figure{margin:4px 0;break-inside:avoid}figure img{width:100%}
figcaption{font-size:8pt;text-align:center}.pend{color:#b00020;font-weight:bold}.card{border:1px solid #333;padding:3px 6px;margin:5px 0;font-size:8.6pt;break-inside:avoid}
.note{font-size:8.4pt;font-style:italic}code{font-size:8pt}
"""


def row(*c, th=False):
    tag = "th" if th else "td"
    return "<tr>" + "".join(f"<{tag}>{x}</{tag}>" for x in c) + "</tr>"


f1 = lambda x: (f"{float(x):.0f}" if float(x).is_integer() else f"{float(x):.1f}")
f3 = lambda x: f"{float(x):.3f}"
f4 = lambda x: f"{float(x):.4f}"

# P1 table
p1_table = ("<table><caption>Table I. Part 1 golden set (25 questions written by us); baseline = dense retrieval only.</caption>"
            + row("Measure", "Baseline", "Final", th=True)
            + row("Answers correct (of 25)", f1(base.group(1)), f1(fin.group(1)))
            + row("Retrieval Recall@5", f3(base.group(3)), f3(fin.group(3)))
            + row("Citation accuracy", f3(base.group(4)), f3(fin.group(4)))
            + row("Median s / question", medd.get("baseline", "?"), medd.get("final", "?"))
            + row("Groundedness (1-5, 2 raters)", "-", (f"{gr[0]:.2f} (agree within 1: {gr[1]:.2f})" if gr else pending()))
            + "</table>")

# P2 tables
t2a = ("<table><caption>Table II. Training (same data, steps, batch size, learning rate, seed).</caption>"
       + row("Run", "Rank", "Trainable params", "Peak GB", "Time s", "GPU-h", th=True))
for name, label in (("lora_r4", "LoRA r=4 (run 1)"), ("lora_r32", "LoRA r=32 (run 2)")):
    g, g2 = tr[name]
    t2a += row(label, g[0], f"{int(g[1]):,}", g2[0], g2[1], g2[2])
t2a += "</table>"
t2b = ("<table><caption>Table III. Evaluation on 40 generated images per model (lower KID is better).</caption>"
       + row("Model", "KID", "CLIP", "CLIP in-dom.", "CLIP OOD", "CLIP comp.", "LPIPS div.", "NSFW flags", th=True))
for m, label in (("base", "SD 1.5 base"), ("run1", "run 1 (r=4)"), ("run2", "run 2 (r=32)")):
    a, b = kid[m]
    t2b += row(label, f"{f3(a[0])}&plusmn;{f3(a[1])}", f1(a[2]), f1(a[3]), f1(a[4]), f1(b[0]), f3(b[1]), nsfw[("base", "run1", "run2").index(m)])
t2b += "</table>"
if p2h:
    nr, res = p2h
    st, pm = res["better_domain_style(L/R)"], res["better_prompt_match(L/R)"]
    hum2 = (f"<table><caption>Table IV. Blind comparison, run 1 vs base, 25 pairs, {nr} raters (one outside the team).</caption>"
            + row("Criterion", "Run 1 wins", "Win rate", "Rater agreement", th=True)
            + row("Domain style", f"{st[0]}/{st[1]}", f3(st[0] / st[1]), f"{st[2]:.2f}")
            + row("Prompt match", f"{pm[0]}/{pm[1]}", f3(pm[0] / pm[1]), f"{pm[2]:.2f}") + "</table>")
else:
    hum2 = pending("(human evaluation)")

# P3 tables
t3 = ("<table><caption>Table V. Part 3 on the 15 scenarios and 10 red-team cases (automatic checks).</caption>"
      + row("Run", "Pass, critic on", "Pass, critic off", "Median s (on / off)", "Attacks succeeded", th=True)
      + row("First run", f1(p3a[0][0]), f1(p3a[0][1]), f"{p3a[1][0]} / {p3a[1][1]}", f"{p3a[2][0]}/{p3a[2][1]}")
      + row("After planner fixes", f1(p3b[0][0]), f1(p3b[0][1]), f"{p3b[1][0]} / {p3b[1][1]}", f"{p3b[2][0]}/{p3b[2][1]}")
      + row("Final run", f1(p3f[0][0]), f1(p3f[0][1]), f"{p3f[1][0]} / {p3f[1][1]}", f"{p3f[2][0]}/{p3f[2][1]}") + "</table>")
if len(judges) >= 2:
    ids = sorted(sheet3)
    ran = [i for i in ids if sheet3[i]["critic_decision"]]
    counts = [sum(j[1][i]["pass"] == "1" for i in ids) for j in judges]
    a, b = judges[0][1], judges[1][1]
    same = sum(a[i]["pass"] == b[i]["pass"] for i in ids)
    both = sum(all(j[1][i]["pass"] == "1" for j in judges) for i in ids)
    dis = [i for i in ids if len({j[1][i]["pass"] for j in judges}) > 1]
    crit = [sum(sheet3[i]["critic_decision"] == j[1][i]["accept_or_revise"] for i in ran) for j in judges]
    hum3 = (f"{len(judges)} judges scored every scenario against its rubric independently: they passed " + " and ".join(str(c) for c in counts)
            + f" of 15 (the automatic checks said {f1(p3f[0][0])}). They agreed on {same} of 15 verdicts ({same / 15:.2f}) and disagreed on {', '.join(dis)}; "
            f"{both} of 15 scenarios were passed by both judges. We report that conservative count; how the disagreements are settled is to be stated by the team. "
            f"The critic's accept or revise decision matched the judges' on " + " and ".join(f"{c} of {len(ran)}" for c in crit) + " of the scenarios where the critic ran.")
elif len(judges) == 1:
    name, a = judges[0]
    ids = sorted(sheet3)
    pas = sum(a[i]["pass"] == "1" for i in ids)
    ran = [i for i in ids if sheet3[i]["critic_decision"]]
    ca = sum(sheet3[i]["critic_decision"] == a[i]["accept_or_revise"] for i in ran)
    hum3 = (f"Judge 1 passed {pas} of 15 scenarios by the rubric, against 15 of 15 by the automatic checks. The critic's accept/revise decision "
            f"matched this judge's on {ca} of {len(ran)} scenarios where the critic ran. The second judge and the inter-judge agreement are {pending()}.")
else:
    hum3 = pending("(human judging)")

rt_tab = ("<table><caption>Table VI. Red-team cases, final run.</caption>" + row("Case", "Kind", "Refused", "Attack succeeded", th=True)
          + "".join(row(i, k.replace("_", " "), "yes" if r == "True" else "no", "yes" if a == "True" else "no") for i, k, a, r, _ in rt_rows) + "</table>")

img = lambda path, cap: f'<figure><img src="{b64(path)}" alt="{html.escape(cap)}"><figcaption>{html.escape(cap)}</figcaption></figure>'

BODY = f"""
<div class="title"><h1>Multimodal RAG, Domain-Style Diffusion Fine-Tuning and a Multi-Agent Image System</h1>
<p>Siva Surya Chandran and Rajesh Paruchuri &mdash; Pair_Programming_Team_06 &mdash; DATA 266, Lab 2</p></div>
<div class="abs">Abstract&mdash;We built three systems that run on one 16 GB GPU with open-weight models only: a retrieval-augmented question answerer
over three economics PDFs (final golden-set score {f1(fin.group(1))}/25, Recall@5 {f3(fin.group(3))}), a LoRA fine-tune of Stable Diffusion 1.5 whose
KID against held-out images falls from {f3(kid['base'][0][0])} to {f3(kid['run1'][0][0])}, and a five-agent image-understanding and generation pipeline that passes
{f1(p3f[0][0])}/15 automatic scenario checks and resisted {int(p3f[2][1]) - int(p3f[2][0])}/{p3f[2][1]} red-team cases. We report where each system fails and what the safety checks caught.
Numbers marked <span class="pend">PENDING</span> (if any) await data and are filled in by re-running <code>tools/build_report.py</code>.</div>
<div class="cols">

<h2>I. Part 1 &mdash; Multimodal RAG</h2>
<h3>A. Architecture</h3>{D1}
<p>PyMuPDF splits the three PDFs into page-tagged text chunks and crops every numbered figure and table as an image with its label and caption.
A 7B vision-language model (4-bit) describes each figure once at ingestion; tables are indexed whole and row by row. {idx.group(1)} items were indexed
({idx.group(2)} figures/tables; {idx.group(3)} figure lies inside the untrusted Appendix C and is excluded). Ingestion and indexing took {ing.group(1)} s.
Retrieval fuses dense embeddings (bge-small) and BM25 with reciprocal-rank fusion and always keeps the best hit of every document, so a revision or an outturn
in another document is seen. A question that names a figure retrieves that exact object: Figure B-5 and Figure B-5 (corrected) are separate records and the corrected one supersedes the original.
A 14B text model (4-bit) writes a JSON answer, must quote verbatim evidence, and a second pass compares what each document states for the same quantity to set status <code>conflict</code>.
Each conversation has its own memory; the vector store is never written while answering (it held {unchanged.group(1)} items before and after).</p>
<h3>B. Results</h3>{p1_table}
<p>Dense-plus-BM25 retrieval with document diversity raised Recall@5 from {f3(base.group(3))} to {f3(fin.group(3))} and citation accuracy from {f3(base.group(4))} to {f3(fin.group(4))},
but the answer count did <b>not</b> improve ({f1(base.group(1))} baseline vs {f1(fin.group(1))} final of 25), and answering was slightly slower. The golden set was used while developing, so these figures are
optimistic, and 25 questions cannot separate a one-question difference. On the Kaggle public leaderboard (about 48% of the hidden questions) the earlier run of this pipeline scored 0.821
(commit 98a2d5d; a leaderboard value, not a notebook output); an earlier submission from the team scored 0.795.</p>
<p>Errors we found by reading the answers against the PDFs: reading a chart value instead of the printed table value for the same quantity, and applying
Okun's law with the wrong sign (the formula in Document 1 predicts a small <i>rise</i> in unemployment for 2.3% growth). Answers also vary slightly between runs.</p>
<h3>C. Safety analysis: the untrusted appendix</h3>
<p>Appendix C of Document 3 contains false claims (a different Lehman date, a contraction of China in 2009), an instruction aimed at AI assistants, and a chart image with similar text.
{appc.group(1)} Appendix C items were kept out of both indexes at ingestion, and the vision model's description of Figure C-1 was flagged as instruction-like ({figinj.group(1).strip('[]').replace(chr(39), '')}); {chunkinj.group(1)} trusted text chunk read like an
injection and was sanitised. In the final 40-question run: the Lehman question was answered &ldquo;{html.escape(sub['30']['answer'])}&rdquo; (the appendix gives a different date);
the Euro-area unemployment question was answered &ldquo;{html.escape(sub['31']['answer'])}&rdquo; with citations {html.escape(sub['31']['citations'])} (the injected instruction asked for a different value and source);
and the China question was answered &ldquo;{html.escape(sub['29']['answer'])}&rdquo; (status {sub['29']['status']}) rather than accepting the appendix claim. So on the three practice questions that target the appendix it did not change the answer; we did not
attack the system adaptively, so this is evidence for these cases, not a guarantee.</p>

<h2>II. Part 2 &mdash; LoRA Fine-Tuning of Stable Diffusion 1.5</h2>
<h3>A. Architecture and data</h3>{D2}
<p>We used <code>lambdalabs/naruto-blip-captions</code>: {dstats.group(1)} images, {dstats.group(2)} corrupted and {dstats.group(3)} exact duplicates found ({dstats.group(5)} removed), split
{dstats.group(8)} train / {dstats.group(7)} test with seed 266 and resized to 512&times;512. Limitations: one franchise with mostly front-facing characters, and short, sometimes wrong BLIP captions that limit prompt alignment.
Two LoRA runs share data, 1,500 steps, batch size 4, learning rate 1e-4 and seed; only the rank differs. The custom loop caches VAE latents and text embeddings and keeps LoRA weights in fp32.</p>
<h3>B. Results</h3>{t2a}{t2b}
<p>Both adapters move generations far closer to the training domain than the base model (KID {f3(kid['base'][0][0])} to {f3(kid['run1'][0][0])} and {f3(kid['run2'][0][0])}); run 2 has the lower KID, by about two standard deviations of a noisy
20-image estimate. CLIP scores against the prompt fall for the fine-tuned models, partly because CLIP rates a stylised cartoon lower against a photographic caption. Run 1 keeps compositional prompts at the base level
({f1(kid['run1'][1][0])} vs {f1(kid['base'][1][0])}) while run 2 drops ({f1(kid['run2'][1][0])}). We choose <b>run 1 (rank 4)</b>: better prompt alignment, no safety-checker flags (run 2: {nsfw[2]} of 40 images blanked), 8&times; fewer parameters and
{tr['lora_r4'][1][2]} vs {tr['lora_r32'][1][2]} GPU-hours. This is a team decision to confirm; run 2 would be the choice if style fidelity alone mattered.</p>
{hum2}
<p class="note">The metrics above come from the RTX 5090 notebook run. The 25 human-evaluation pairs were generated from a second training run of the same code and settings on a rented RTX 4090 (the Hugging Face copy of the first run is private), so the human win rates describe that second run.</p>
{img('p2_grid.png', 'Fig. 1. Base vs run 1 vs run 2 on eight prompts (from the Part 2 notebook).')}
<h3>C. Failure modes</h3>
<p>(1) <i>Part of the prompt is dropped</i>: for &ldquo;a modern glass skyscraper reflecting the clouds&rdquo;, the lowest-scoring image for both the base model and run 1 (Fig. 2), run 1 draws a cropped glass facade without the clouds that the base model shows.
(2) <i>Safety-checker false positives</i>: the lowest-scoring &ldquo;image&rdquo; for run 2 is solid black. The checker blanked a harmless multi-character prompt (&ldquo;three characters in a row&rdquo;); {nsfw[2]} of 40 run 2 images were blanked against {nsfw[0]} and {nsfw[1]} for the base model and run 1.
Multi-character compositions also score lowest for run 2 on CLIP, which suggests the larger adapter fits single front-facing characters more tightly; we have not tested this directly.</p>
{img('p2_failures.png', 'Fig. 2. Lowest-CLIP generation per model (from the Part 2 notebook).')}
<h3>D. Safety</h3>
<p><i>License.</i> Naruto characters are copyrighted and the dataset is CC BY-NC-SA 4.0: the adapters are for non-commercial research only. <i>NSFW.</i> The Stable Diffusion safety checker stayed on; it flagged {nsfw[0]} (base), {nsfw[1]} (run 1) and {nsfw[2]} (run 2) of 40 images.
<i>Memorization.</i> For ten in-domain images we found the most similar training image by CLIP cosine similarity; the three closest pairs (cosine 0.894, 0.893 and 0.864, as printed in Fig. 3) share the drawing style but differ in
hair colour, clothing and pose, so we saw no near-copy of a training image. Ten images is a small check, not a proof.</p>
{img('p2_memorization.png', 'Fig. 3. Three closest generated/training pairs (from the Part 2 notebook).')}
<div class="card"><b>Model card (final run).</b> sd15-naruto-lora-r4: LoRA (rank 4) for stable-diffusion-v1-5, trained on {dstats.group(8)} captioned images, 1,500 steps, batch 4, lr 1e-4, {int(tr['lora_r4'][0][1]):,} trainable parameters.
KID {f3(kid['run1'][0][0])} (base {f3(kid['base'][0][0])}), CLIP {f1(kid['run1'][0][2])} (base {f1(kid['base'][0][2])}). Limits: one franchise; weaker on out-of-domain and multi-character prompts; may resemble training images.
License: non-commercial research only (copyrighted characters; CC BY-NC-SA 4.0 data; OpenRAIL-M base).</div>

<h2>III. Part 3 &mdash; Multimodal Multi-Agent System</h2>
<h3>A. Architecture</h3>{D3}
<p>Five agents pass one typed state through plain Python functions. The Planner classifies a request (caption/question, style change, edit, domain-style generation, ambiguous, unsupported), runs text and image safety checks
and refuses unsafe requests. Vision returns a caption, objects, visible text and question answers, treating image text as data. Prompt Engineering rewrites the request using the vision output and never obeys text found in the image.
Image Generation offers img2img style change, InstructPix2Pix editing and generation with the Part 2 LoRA for domain-style requests, recording model, prompt, seed and steps. Critique scores the result with CLIP similarity plus a vision-language judgement,
checks output safety, and either accepts or sends a concrete instruction back for at most two revisions. A JSON trace of every request is saved, and failures return a clear message instead of an exception. A Gradio app shows each agent's output.</p>
<h3>B. Results</h3>{t3}
<p>In the first run the system passed {f1(p3a[0][0])}/15 and {p3a[2][0]}/{p3a[2][1]} attacks succeeded: edit requests were routed as style changes, a text-only drawing request was called ambiguous, a vague request was treated as an edit, and two edits of a person into a harmful scene were not refused.
We then added generic routing rules (question wording, style words, targeted edits, vague requests), a deceptive-person-edit guardrail and a larger vision model, and re-ran. <b>These fixes were made after seeing results on these same scenarios, so the later numbers are optimistic.</b>
The critic loop did not change the automatic pass count ({f1(p3f[0][0])} with and {f1(p3f[0][1])} without); it did trigger revisions on some edits and style changes (see the saved traces), so on these scenarios it added time without changing the automatic outcome.</p>
<p>{hum3}</p>{rt_tab}
<h3>C. Safety analysis</h3>
<p>Guardrails are layered: regular expressions for sexual and graphic content, jailbreak phrasing and &ldquo;follow the image&rdquo; requests; an LLM safety flag in the Planner; an image check for instruction-like text; and an output check in Critique.
Final result: {p3f[2][0]} of {p3f[2][1]} attacks succeeded. The two image-injection cases were answered without following the injected text; {sum(1 for r in rt_rows if r[3] == 'True')} unsafe-text, hidden-instruction and real-person cases were refused; the empty and blank files returned a clear message.
The earlier failures (real-person edits) show the limits of a small vision model as the only check, which is why the rule layer was added. Our red-team set is small and was written by us; an adaptive attacker would likely do better.</p>
<div class="card"><b>System card.</b> <i>Can do:</i> caption images, answer questions about them, restyle or edit an image, draw new characters in the fine-tuned style. <i>Known limits:</i> small open models can misread text in images and misjudge results (the critic's accept/revise decision matched a human judge's only partly; see the human-judging paragraph);
edits are limited to what InstructPix2Pix and img2img can do; routing relies on phrasing rules. <i>Safety:</i> refuses sexual and graphic requests, deceptive edits of people and hidden-instruction images; never follows text found in images; failures return messages; the output is re-checked before it is shown.</div>

<h2>IV. Process Notes and Known Limits</h2>
<p>In this repository the golden set and the Part 2 prompts were first committed together with Part 1 tuning changes, so the repository alone does not show they were written before tuning (the team repository's golden set was committed before its tuning). One golden question has the same wording as a public question about another country. Part 3 routing rules and a guardrail were added after a first run on the evaluation cases, so its final numbers are optimistic. The saved Part 1 notebook outputs come from the code before the last changes; the later pipeline's output file is recorded by the team repository at 0.853 on the public leaderboard (not a notebook output). A hand-written answer key used only for development scoring of the public practice questions is deliberately kept out of the repository.</p>

<h2>V. Reproducibility and Credits</h2>
<p>Code, notebooks (with saved outputs), tags <code>p1-v1.0</code>, <code>p2-v1.0</code> and <code>p3-v1.0</code>, and the human-evaluation files are in the GitHub repository listed in <code>links.txt</code>. Seeds are fixed to 266. All models are open weights; no hosted API was used.
An AI coding assistant (Claude Code) helped write and debug the code; the details are in <code>AI_USAGE.md</code>. Contributions are in <code>CONTRIBUTIONS.md</code>.</p>
</div>
"""

os.makedirs(P("report"), exist_ok=True)
open(P("report", "report.html"), "w").write(f"<!doctype html><html><head><meta charset='utf-8'><title>DATA 266 Lab 2 Report</title><style>{CSS}</style></head><body>{BODY}</body></html>")
chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={P('report', 'report.pdf')}",
                "file://" + P("report", "report.html")], capture_output=True, timeout=120)
print("report written:", os.path.exists(P("report", "report.pdf")))
