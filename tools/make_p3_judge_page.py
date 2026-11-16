"""Builds human_eval_p3/judge.html: independent pass/fail judging of the 15 Part 3 scenarios.

Inputs: scenario definitions (p3_agents/scenarios.py), artifacts/p3/human_eval_p3.csv (outputs of the executed notebook)
and the saved images under artifacts/p3/work_p3. The critic's own accept/revise decision is NOT shown, so each judge's
accept/revise call is independent (critic agreement is computed afterwards).
Each judge enters a name, judges every scenario, and downloads judge_<name>.csv (id, pass, accept_or_revise, note).
"""
import base64
import csv
import json
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src = open(os.path.join(ROOT, "p3_agents", "scenarios.py")).read()
ns = {}
exec(src, ns)
scen = {s["id"]: s for s in ns["SCENARIOS"]}
art = os.path.join(ROOT, "artifacts", "p3")
sheet = {r["id"]: r for r in csv.DictReader(open(os.path.join(art, "human_eval_p3.csv")))}


def jpg(path):
    if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
        return ""
    out = tempfile.mktemp(suffix=".jpg")
    r = subprocess.run(["sips", "-s", "format", "jpeg", "-s", "formatOptions", "70", "-Z", "512", path, "--out", out],
                       capture_output=True)
    if r.returncode != 0 or not os.path.exists(out):
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(open(out, "rb").read()).decode()


items = []
for sid in sorted(sheet):
    s, r = scen[sid], sheet[sid]
    items.append({"id": sid, "cat": s["cat"], "request": s["request"], "rubric": s["rubric"],
                  "input": jpg(os.path.join(art, s["image"])) if s.get("image") else "",
                  "output": jpg(os.path.join(art, r["output"])) if r.get("output") else "",
                  "answer": r.get("answer", ""), "message": r.get("message", "")})

PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Scenario Judging</title><style>
:root{--bg:#fafaf8;--fg:#1c1c1a;--mut:#6b6b66;--card:#fff;--line:#dcdcd5;--acc:#2f5d8a;--accfg:#fff;--ok:#2d7a4d}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#171716;--fg:#ececE8;--mut:#a3a39c;--card:#212120;--line:#3a3a37;--acc:#7fb0e0;--accfg:#10202f;--ok:#6cc08b}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:980px;margin:0 auto;padding:20px 16px 90px}h1{font-size:1.35rem;margin:0 0 4px}p.sub{color:var(--mut);margin:0 0 16px}
.bar{position:sticky;top:0;background:var(--bg);padding:10px 0;border-bottom:1px solid var(--line);z-index:2;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
.prog{flex:1;min-width:140px;height:8px;background:var(--line);border-radius:6px;overflow:hidden}.prog i{display:block;height:100%;background:var(--acc);width:0}
input[type=text],textarea{padding:8px 10px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--fg);font:inherit}
textarea{width:100%;margin-top:8px}button{font:inherit;padding:9px 16px;border-radius:8px;border:1px solid var(--acc);background:var(--acc);color:var(--accfg);cursor:pointer}
button.ghost{background:transparent;color:var(--acc)}button:disabled{opacity:.45;cursor:not-allowed}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px;margin:16px 0}.card.done{border-color:var(--ok)}
.card h2{font-size:.95rem;margin:0 0 6px;color:var(--mut);font-weight:600}.req{font-weight:600;margin:0 0 6px}.rub{margin:0 0 10px;color:var(--mut)}
.imgs{display:flex;gap:12px;flex-wrap:wrap}.imgs figure{margin:0;flex:1;min-width:220px}.imgs img{width:100%;height:auto;border-radius:8px;display:block}
figcaption{font-size:.8rem;color:var(--mut);margin-top:3px}.txt{background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:8px 10px;margin:8px 0}
.q{margin:10px 0 0}.q b{display:block;margin-bottom:6px;font-weight:600}.opts{display:flex;gap:10px;flex-wrap:wrap}
.opts label{flex:1;min-width:110px;text-align:center;border:1px solid var(--line);border-radius:8px;padding:9px;cursor:pointer}
.opts input{position:absolute;opacity:0}.opts label:has(input:checked){background:var(--acc);color:var(--accfg);border-color:var(--acc)}
.opts label:has(input:focus-visible){outline:2px solid var(--acc);outline-offset:2px}.msg{color:var(--mut);font-size:.9rem}
</style></head><body><main>
<h1>Judge the 15 scenarios</h1>
<p class="sub">For each scenario read the request and the rubric, look at the input and the system's output (or its text reply), and decide independently: (1) does the result <b>pass the rubric</b>? (2) would <b>you</b> accept it or ask for a revision? Judge on your own; agreement between two judges is measured afterwards. Answers are saved in this browser per judge name.</p>
<div class="bar"><input type="text" id="name" placeholder="Your name" aria-label="Your name"><div class="prog" aria-hidden="true"><i id="pi"></i></div>
<span id="pt" class="msg">0 / 15</span><button id="dl" disabled>Download my judgments (CSV)</button><button id="rs" class="ghost" type="button">Clear my answers</button></div>
<div id="list"></div><p class="msg" id="foot"></p></main>
<script>
const ITEMS = __ITEMS__; const KEY = "p3_judge_v1";
let st = {}; try { st = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { st = {}; }
st.all = st.all || {}; st.name = st.name || "";
const nk = () => st.name.trim().toLowerCase(); const cur = () => st.all[nk()] || {};
function setA(id, k, v){ const n = nk(); if (!n) return; st.all[n] = st.all[n] || {}; st.all[n][id] = st.all[n][id] || {}; st.all[n][id][k] = v; }
function save(){ try { localStorage.setItem(KEY, JSON.stringify(st)); } catch (e) {} }
const list = document.getElementById("list");
ITEMS.forEach(p => {
  const c = document.createElement("div"); c.className = "card"; c.id = "c" + p.id;
  const grp = (k, label, a, b) => `<div class="q"><b>${label}</b><div class="opts"><label><input type="radio" name="${k}${p.id}" value="${a[0]}"> ${a[1]}</label><label><input type="radio" name="${k}${p.id}" value="${b[0]}"> ${b[1]}</label></div></div>`;
  c.innerHTML = `<h2>${p.id} &middot; ${p.cat}</h2><p class="req"></p><p class="rub"></p><div class="imgs"></div><div class="txt" hidden></div>
    ${grp("p", "Does the result pass the rubric?", ["1", "Pass"], ["0", "Fail"])}${grp("a", "Would you accept it or ask for a revision?", ["accept", "Accept"], ["revise", "Revise"])}
    <textarea rows="2" placeholder="Optional note (why pass/fail)"></textarea>`;
  c.querySelector(".req").textContent = "Request: " + p.request; c.querySelector(".rub").textContent = "Rubric: " + p.rubric;
  const imgs = c.querySelector(".imgs");
  [["Input", p.input], ["Output", p.output]].forEach(([cap, src]) => { if (!src) return; const f = document.createElement("figure"); f.innerHTML = `<img alt="${cap} image"><figcaption>${cap}</figcaption>`; f.querySelector("img").src = src; imgs.appendChild(f); });
  const t = c.querySelector(".txt"); const text = [p.answer && ("Answer: " + p.answer), p.message && ("System message: " + p.message)].filter(Boolean).join("\n");
  if (text) { t.hidden = false; t.style.whiteSpace = "pre-wrap"; t.textContent = text; }
  if (!p.output && !text) { t.hidden = false; t.textContent = "(no output image and no text reply was returned)"; }
  list.appendChild(c);
  c.addEventListener("change", e => { const x = e.target; if (x.type !== "radio") return; if (!nk()) { alert("Enter your name first."); x.checked = false; return; } setA(p.id, x.name[0], x.value); save(); refresh(); });
  c.querySelector("textarea").addEventListener("input", e => { if (!nk()) return; setA(p.id, "n", e.target.value); save(); });
});
const nameEl = document.getElementById("name"); nameEl.value = st.name;
function paint(){ ITEMS.forEach(p => { const a = cur()[p.id] || {};
  ["p", "a"].forEach(k => document.querySelectorAll(`input[name=${k}${p.id}]`).forEach(r => { r.checked = a[k] === r.value; }));
  document.querySelector(`#c${p.id} textarea`).value = a.n || ""; }); refresh(); }
nameEl.addEventListener("input", () => { st.name = nameEl.value; save(); paint(); });
function refresh(){ let d = 0; ITEMS.forEach(p => { const a = cur()[p.id] || {}; const ok = a.p !== undefined && a.a; if (ok) d++; document.getElementById("c" + p.id).classList.toggle("done", !!ok); });
  document.getElementById("pt").textContent = d + " / " + ITEMS.length; document.getElementById("pi").style.width = (100 * d / ITEMS.length) + "%";
  const ready = d === ITEMS.length && nk().length > 0; document.getElementById("dl").disabled = !ready;
  document.getElementById("foot").textContent = ready ? "All scenarios judged. Download your CSV and send it back." : "Judge every scenario and enter your name to enable the download."; }
document.getElementById("rs").addEventListener("click", () => { if (nk() && confirm("Clear all answers for " + st.name + "?")) { delete st.all[nk()]; save(); paint(); } });
document.getElementById("dl").addEventListener("click", () => {
  const q = s => '"' + String(s || "").replace(/"/g, '""') + '"'; const rows = ["id,pass,accept_or_revise,note"];
  ITEMS.forEach(p => { const a = cur()[p.id]; rows.push([p.id, a.p, a.a, q(a.n)].join(",")); });
  const blob = new Blob([rows.join("\n") + "\n"], {type: "text/csv"}); const url = URL.createObjectURL(blob); const l = document.createElement("a");
  l.href = url; l.download = "judge_" + st.name.trim().replace(/[^A-Za-z0-9_-]+/g, "_") + ".csv"; l.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); });
paint();
</script></body></html>
"""
os.makedirs(os.path.join(ROOT, "human_eval_p3"), exist_ok=True)
open(os.path.join(ROOT, "human_eval_p3", "judge.html"), "w").write(PAGE.replace("__ITEMS__", json.dumps(items)))
print("wrote human_eval_p3/judge.html with", len(items), "scenarios;", sum(1 for i in items if i["output"]), "with output images")
