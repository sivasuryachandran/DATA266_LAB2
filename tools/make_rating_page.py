"""Builds human_eval_p2/rate.html: a self-contained blind rating page for the Part 2 human evaluation.

Inputs (from p2_finetune.human_eval.build_sheet): human_eval_p2/jpg/pair_XX.jpg and rating_sheet_TEMPLATE.csv.
The page never shows which side is the fine-tuned model (that key stays in KEY_do_not_share.csv).
Each rater opens the page, enters a name, rates all pairs and downloads rater_<name>.csv.
Then:  from p2_finetune.human_eval import win_rate; win_rate("human_eval_p2", [<rater csv files>])
"""
import base64
import csv
import json
import os
import sys

root = sys.argv[1] if len(sys.argv) > 1 else "human_eval_p2"
pairs = []
for row in csv.DictReader(open(os.path.join(root, "rating_sheet_TEMPLATE.csv"))):
    i = int(row["pair"])
    b = base64.b64encode(open(os.path.join(root, "jpg", f"pair_{i:02d}.jpg"), "rb").read()).decode()
    pairs.append({"pair": i, "prompt": row["prompt"], "img": "data:image/jpeg;base64," + b})

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Image Pair Rating</title>
<style>
:root{--bg:#fafaf8;--fg:#1c1c1a;--mut:#6b6b66;--card:#fff;--line:#dcdcd5;--acc:#2f5d8a;--accfg:#fff;--ok:#2d7a4d}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#171716;--fg:#ececE8;--mut:#a3a39c;--card:#212120;--line:#3a3a37;--acc:#7fb0e0;--accfg:#10202f;--ok:#6cc08b}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:980px;margin:0 auto;padding:20px 16px 90px}
h1{font-size:1.35rem;margin:0 0 4px}p.sub{color:var(--mut);margin:0 0 16px}
.bar{position:sticky;top:0;background:var(--bg);padding:10px 0;border-bottom:1px solid var(--line);z-index:2;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
.prog{flex:1;min-width:140px;height:8px;background:var(--line);border-radius:6px;overflow:hidden}.prog i{display:block;height:100%;background:var(--acc);width:0}
input[type=text]{padding:8px 10px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--fg);font:inherit;min-width:160px}
button{font:inherit;padding:9px 16px;border-radius:8px;border:1px solid var(--acc);background:var(--acc);color:var(--accfg);cursor:pointer}
button.ghost{background:transparent;color:var(--acc)}button:disabled{opacity:.45;cursor:not-allowed}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px;margin:16px 0}
.card.done{border-color:var(--ok)}
.card h2{font-size:.95rem;margin:0 0 4px;color:var(--mut);font-weight:600}.card .prompt{font-weight:600;margin:0 0 10px}
.card img{width:100%;height:auto;border-radius:8px;display:block}
.cols{display:flex;justify-content:space-around;color:var(--mut);font-size:.85rem;margin:4px 0 10px}
.q{margin:10px 0 0}.q b{display:block;margin-bottom:6px;font-weight:600}
.opts{display:flex;gap:10px;flex-wrap:wrap}.opts label{flex:1;min-width:110px;text-align:center;border:1px solid var(--line);border-radius:8px;padding:9px;cursor:pointer}
.opts input{position:absolute;opacity:0}.opts label:has(input:checked){background:var(--acc);color:var(--accfg);border-color:var(--acc)}
.opts label:has(input:focus-visible){outline:2px solid var(--acc);outline-offset:2px}
.msg{color:var(--mut);font-size:.9rem}
</style></head><body><main>
<h1>Which image is better?</h1>
<p class="sub">Each pair shows two images made from the same prompt. Which side is which model is hidden. For each pair choose the <b>left</b> or <b>right</b> image for (1) domain style (looks like the Naruto anime-style training data) and (2) prompt match (shows what the prompt describes). Your answers are saved in this browser as you go.</p>
<div class="bar">
  <input type="text" id="name" placeholder="Your name" aria-label="Your name">
  <div class="prog" aria-hidden="true"><i id="pi"></i></div>
  <span id="pt" class="msg">0 / 25</span>
  <button id="dl" disabled>Download my ratings (CSV)</button>
  <button id="rs" class="ghost" type="button">Clear my answers</button>
</div>
<div id="list"></div>
<p class="msg" id="foot"></p>
</main>
<script>
const PAIRS = __PAIRS__;
const KEY = "p2_human_eval_v1";
let st = {};
try { st = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { st = {}; }
st.all = st.all || {}; st.name = st.name || "";
const nk = () => st.name.trim().toLowerCase();
function cur(){ return st.all[nk()] || {}; }
function setAns(pair, k, v){ const n = nk(); if (!n) return; st.all[n] = st.all[n] || {}; st.all[n][pair] = st.all[n][pair] || {}; st.all[n][pair][k] = v; }
function save(){ try { localStorage.setItem(KEY, JSON.stringify(st)); } catch (e) {} }
const list = document.getElementById("list");
PAIRS.forEach(p => {
  const c = document.createElement("div"); c.className = "card"; c.id = "c" + p.pair;
  const group = (k, label) => `<div class="q"><b>${label}</b><div class="opts">
     <label><input type="radio" name="${k}${p.pair}" value="L"> Left</label>
     <label><input type="radio" name="${k}${p.pair}" value="R"> Right</label></div></div>`;
  c.innerHTML = `<h2>Pair ${p.pair + 1} of ${PAIRS.length}</h2><div class="prompt"></div>
    <img alt="Two generated images, left and right" src="${p.img}">
    <div class="cols"><span>Left</span><span>Right</span></div>
    ${group("s", "1. Better domain style (anime / Naruto look)")}${group("m", "2. Better prompt match")}`;
  c.querySelector(".prompt").textContent = "Prompt: " + p.prompt;
  list.appendChild(c);
  c.addEventListener("change", e => {
    const t = e.target; if (t.type !== "radio") return;
    if (!nk()) { alert("Enter your name first."); t.checked = false; return; }
    setAns(p.pair, t.name[0], t.value); save(); refresh();
  });
});
const nameEl = document.getElementById("name"); nameEl.value = st.name;
function paint(){
  PAIRS.forEach(p => { const a = cur()[p.pair] || {};
    ["s", "m"].forEach(k => document.querySelectorAll(`input[name=${k}${p.pair}]`).forEach(r => { r.checked = a[k] === r.value; })); });
  refresh();
}
nameEl.addEventListener("input", () => { st.name = nameEl.value; save(); paint(); });
function refresh(){
  let done = 0;
  PAIRS.forEach(p => { const a = cur()[p.pair] || {}; const ok = a.s && a.m; if (ok) done++; document.getElementById("c" + p.pair).classList.toggle("done", !!ok); });
  document.getElementById("pt").textContent = done + " / " + PAIRS.length;
  document.getElementById("pi").style.width = (100 * done / PAIRS.length) + "%";
  const ready = done === PAIRS.length && st.name.trim().length > 0;
  document.getElementById("dl").disabled = !ready;
  document.getElementById("foot").textContent = ready ? "All pairs rated. Download your CSV and send it back." : "Rate every pair and enter your name to enable the download.";
}
document.getElementById("rs").addEventListener("click", () => { if (nk() && confirm("Clear all answers for " + st.name + "?")) { delete st.all[nk()]; save(); paint(); } });
document.getElementById("dl").addEventListener("click", () => {
  const q = s => '"' + String(s).replace(/"/g, '""') + '"';
  const rows = ["pair,prompt,better_domain_style(L/R),better_prompt_match(L/R)"];
  PAIRS.forEach(p => { const a = cur()[p.pair]; rows.push([p.pair, q(p.prompt), a.s, a.m].join(",")); });
  const blob = new Blob([rows.join("\n") + "\n"], {type: "text/csv"});
  const url = URL.createObjectURL(blob); const a = document.createElement("a");
  a.href = url; a.download = "rater_" + st.name.trim().replace(/[^A-Za-z0-9_-]+/g, "_") + ".csv"; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
paint();
</script></body></html>
"""
open(os.path.join(root, "rate.html"), "w").write(PAGE.replace("__PAIRS__", json.dumps(pairs)))
print("wrote", os.path.join(root, "rate.html"), len(pairs), "pairs")
