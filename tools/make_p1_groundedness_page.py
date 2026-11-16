"""Builds human_eval_p1/rate.html: groundedness rating (1-5) of the 25 final golden-set answers.

For each answer the page shows the question, the system's answer and status, and the text of every cited source
(page text or the table/figure record from the index). The gold answer is NOT shown: the rater judges only whether
every claim in the answer is supported by the cited sources. Each rater downloads rater_<name>.csv (id, groundedness, note).
"""
import ast
import csv
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
res = list(csv.DictReader(open(os.path.join(ROOT, "artifacts", "p1", "golden_results_final.csv"))))
gold = {g["id"]: g for g in json.load(open(os.path.join(ROOT, "golden_set.json"), encoding="utf-8-sig"))}
items = json.load(open(os.path.join(ROOT, "artifacts", "p1", "work_p1", "items.json")))
by_id = {i["id"]: i for i in items}


def source_text(cite):
    doc, _, rest = cite.partition(":")
    if rest.startswith("p") and rest[1:].isdigit():
        txt = "\n".join(i["text"] for i in items if i["doc_id"] == doc and str(i["page"]) == rest[1:] and i["kind"] == "text")
        return txt
    hits = [i for i in items if i["doc_id"] == doc and i["figure_id"] == rest and i["kind"] in ("table", "figure")]
    return "\n".join(i["text"] for i in hits)


out = []
for r in res:
    try:
        cites = ast.literal_eval(r["citations"])
    except Exception:
        cites = []
    out.append({"id": r["id"], "question": gold[r["id"]]["question"], "answer": r["answer"], "status": r["status"],
                "sources": [{"id": c, "text": source_text(c)[:2500]} for c in cites]})

PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Groundedness Rating</title><style>
:root{--bg:#fafaf8;--fg:#1c1c1a;--mut:#6b6b66;--card:#fff;--line:#dcdcd5;--acc:#2f5d8a;--accfg:#fff;--ok:#2d7a4d}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#171716;--fg:#ececE8;--mut:#a3a39c;--card:#212120;--line:#3a3a37;--acc:#7fb0e0;--accfg:#10202f;--ok:#6cc08b}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:980px;margin:0 auto;padding:20px 16px 90px}h1{font-size:1.35rem;margin:0 0 4px}p.sub{color:var(--mut);margin:0 0 16px}
.bar{position:sticky;top:0;background:var(--bg);padding:10px 0;border-bottom:1px solid var(--line);z-index:2;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
.prog{flex:1;min-width:140px;height:8px;background:var(--line);border-radius:6px;overflow:hidden}.prog i{display:block;height:100%;background:var(--acc);width:0}
input[type=text],textarea{padding:8px 10px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--fg);font:inherit}textarea{width:100%;margin-top:8px}
button{font:inherit;padding:9px 16px;border-radius:8px;border:1px solid var(--acc);background:var(--acc);color:var(--accfg);cursor:pointer}
button.ghost{background:transparent;color:var(--acc)}button:disabled{opacity:.45;cursor:not-allowed}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px;margin:16px 0}.card.done{border-color:var(--ok)}
.card h2{font-size:.95rem;margin:0 0 6px;color:var(--mut);font-weight:600}.qq{font-weight:600;margin:0 0 6px}.ans{background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:8px 10px;margin:8px 0}
details{border:1px solid var(--line);border-radius:8px;padding:6px 10px;margin:6px 0}summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;font:inherit;font-size:.9rem;margin:6px 0 0;max-height:320px;overflow:auto}
.opts{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}.opts label{flex:1;min-width:56px;text-align:center;border:1px solid var(--line);border-radius:8px;padding:9px;cursor:pointer}
.opts input{position:absolute;opacity:0}.opts label:has(input:checked){background:var(--acc);color:var(--accfg);border-color:var(--acc)}
.opts label:has(input:focus-visible){outline:2px solid var(--acc);outline-offset:2px}.msg{color:var(--mut);font-size:.9rem}
</style></head><body><main>
<h1>How well is each answer grounded in its cited sources?</h1>
<p class="sub">Score 1-5: <b>5</b> every claim is supported by the cited sources; <b>3</b> mostly supported, some unsupported detail; <b>1</b> the cited sources do not support the answer (or no source is cited for a factual claim). Judge only against the cited sources shown, not outside knowledge. Answers are saved in this browser per rater name.</p>
<div class="bar"><input type="text" id="name" placeholder="Your name" aria-label="Your name"><div class="prog" aria-hidden="true"><i id="pi"></i></div>
<span id="pt" class="msg">0 / 25</span><button id="dl" disabled>Download my ratings (CSV)</button><button id="rs" class="ghost" type="button">Clear my answers</button></div>
<div id="list"></div><p class="msg" id="foot"></p></main>
<script>
const ITEMS = __ITEMS__; const KEY = "p1_ground_v1";
let st = {}; try { st = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { st = {}; }
st.all = st.all || {}; st.name = st.name || "";
const nk = () => st.name.trim().toLowerCase(); const cur = () => st.all[nk()] || {};
function setA(id, k, v){ const n = nk(); if (!n) return; st.all[n] = st.all[n] || {}; st.all[n][id] = st.all[n][id] || {}; st.all[n][id][k] = v; }
function save(){ try { localStorage.setItem(KEY, JSON.stringify(st)); } catch (e) {} }
const list = document.getElementById("list");
ITEMS.forEach(p => {
  const c = document.createElement("div"); c.className = "card"; c.id = "c" + p.id;
  c.innerHTML = `<h2>${p.id} &middot; status: ${p.status}</h2><p class="qq"></p><div class="ans"></div><div class="src"></div>
    <div class="opts">${[1,2,3,4,5].map(n => `<label><input type="radio" name="g${p.id}" value="${n}"> ${n}</label>`).join("")}</div>
    <textarea rows="2" placeholder="Optional note"></textarea>`;
  c.querySelector(".qq").textContent = "Question: " + p.question; c.querySelector(".ans").textContent = "Answer: " + p.answer;
  const s = c.querySelector(".src");
  if (!p.sources.length) { s.innerHTML = '<p class="msg">No sources were cited.</p>'; }
  p.sources.forEach(x => { const d = document.createElement("details"); const sm = document.createElement("summary"); sm.textContent = "Cited source: " + x.id; const pre = document.createElement("pre"); pre.textContent = x.text || "(no text found for this id)"; d.append(sm, pre); s.appendChild(d); });
  list.appendChild(c);
  c.addEventListener("change", e => { const x = e.target; if (x.type !== "radio") return; if (!nk()) { alert("Enter your name first."); x.checked = false; return; } setA(p.id, "g", x.value); save(); refresh(); });
  c.querySelector("textarea").addEventListener("input", e => { if (!nk()) return; setA(p.id, "n", e.target.value); save(); });
});
const nameEl = document.getElementById("name"); nameEl.value = st.name;
function paint(){ ITEMS.forEach(p => { const a = cur()[p.id] || {};
  document.querySelectorAll(`input[name=g${p.id}]`).forEach(r => { r.checked = a.g === r.value; }); document.querySelector(`#c${p.id} textarea`).value = a.n || ""; }); refresh(); }
nameEl.addEventListener("input", () => { st.name = nameEl.value; save(); paint(); });
function refresh(){ let d = 0; ITEMS.forEach(p => { const ok = (cur()[p.id] || {}).g; if (ok) d++; document.getElementById("c" + p.id).classList.toggle("done", !!ok); });
  document.getElementById("pt").textContent = d + " / " + ITEMS.length; document.getElementById("pi").style.width = (100 * d / ITEMS.length) + "%";
  const ready = d === ITEMS.length && nk().length > 0; document.getElementById("dl").disabled = !ready;
  document.getElementById("foot").textContent = ready ? "All answers rated. Download your CSV and send it back." : "Rate every answer and enter your name to enable the download."; }
document.getElementById("rs").addEventListener("click", () => { if (nk() && confirm("Clear all answers for " + st.name + "?")) { delete st.all[nk()]; save(); paint(); } });
document.getElementById("dl").addEventListener("click", () => {
  const q = s => '"' + String(s || "").replace(/"/g, '""') + '"'; const rows = ["id,groundedness,note"];
  ITEMS.forEach(p => { const a = cur()[p.id]; rows.push([p.id, a.g, q(a.n)].join(",")); });
  const blob = new Blob([rows.join("\n") + "\n"], {type: "text/csv"}); const url = URL.createObjectURL(blob); const l = document.createElement("a");
  l.href = url; l.download = "rater_" + st.name.trim().replace(/[^A-Za-z0-9_-]+/g, "_") + ".csv"; l.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); });
paint();
</script></body></html>
"""
os.makedirs(os.path.join(ROOT, "human_eval_p1"), exist_ok=True)
open(os.path.join(ROOT, "human_eval_p1", "rate.html"), "w").write(PAGE.replace("__ITEMS__", json.dumps(out)))
print("wrote human_eval_p1/rate.html;", len(out), "answers;", sum(1 for o in out if not o["sources"]), "without sources;",
      sum(1 for o in out for s in o["sources"] if not s["text"]), "cited ids without text")
