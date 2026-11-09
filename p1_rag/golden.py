"""1.4 Golden set schema, coverage check and the four measurements.

golden_set.json is written BY HAND from the three PDFs (before tuning; commit it). Entry:
 {"id": "G01", "category": "fact|figure|near_identical|two_source|multi_turn|conflict|unanswerable",
  "conversation_id": "G01", "turn": 1, "question": "...",
  "gold_answer": "...", "gold_numbers": [5.2], "gold_keywords": ["..."],
  "gold_status": "answered", "gold_sources": ["DOC1:p12", "DOC2:Figure A-3"],
  "gold_pages": [["DOC1", 12]], "gold_figures": ["Figure A-3"]}
"""
import json
import statistics
from collections import Counter


from p1_rag.text_utils import numbers_match

CATEGORIES = {"fact_table": 5, "figure_near_identical": 6, "two_source": 3, "multi_turn": 5,
              "conflict_revision": 3, "unanswerable": 3}   # = 25


def check_coverage(golden):
    c = Counter(g["category"] for g in golden)
    problems = [f"{k}: have {c.get(k, 0)}, need {v}" for k, v in CATEGORIES.items() if c.get(k, 0) < v]
    convs = Counter(g["conversation_id"] for g in golden if g["category"] == "multi_turn")
    if sum(1 for v in convs.values() if v >= 2) < 2:
        problems.append("multi_turn: need at least 2 conversations with 2+ turns")
    return problems


def is_correct(gold, pred) -> bool:
    if pred["status"] != gold["gold_status"]:
        return False
    if gold.get("gold_numbers"):
        return numbers_match(gold["gold_numbers"], pred["answer"], 0.01)
    kws = gold.get("gold_keywords", [])
    return all(k.lower() in pred["answer"].lower() for k in kws) if kws else True


def run_golden(system, golden):
    """Answers the golden set conversation by conversation; returns per-question records."""
    import time
    recs = []
    by_conv = {}
    for g in golden:
        by_conv.setdefault(g["conversation_id"], []).append(g)
    for conv, qs in by_conv.items():
        chat = system.new_chat()
        for g in sorted(qs, key=lambda x: x["turn"]):
            t0 = time.time()
            pred = chat.answer(g["question"])
            secs = time.time() - t0
            retrieved = system.last_retrieved[:5]
            gp = {tuple(p) for p in g.get("gold_pages", [])}
            gf = set(g.get("gold_figures", []))
            hit = any((r["doc_id"], r["page"]) in gp or (r["figure_id"] and r["figure_id"] in gf) for r in retrieved)
            sup = [c for c in pred["citations"] if c in set(g.get("gold_sources", []))]
            recs.append({"id": g["id"], "category": g["category"], "correct": is_correct(g, pred),
                         "recall_hit": hit if (gp or gf) else None,
                         "cite_total": len(pred["citations"]), "cite_ok": len(sup), "seconds": secs, **pred})
    return recs


def summarize(recs) -> dict:
    df = pd.DataFrame(recs)
    rec = df["recall_hit"].dropna()
    return {"correct_out_of_25": int(df["correct"].sum()), "n": len(df),
            "recall@5": round(float(rec.mean()), 3) if len(rec) else None,
            "citation_accuracy": round(df["cite_ok"].sum() / max(df["cite_total"].sum(), 1), 3),
            "median_sec_per_question": round(statistics.median(df["seconds"]), 2)}


def human_sheet(recs, path="human_eval_p1.csv"):
    """Both members fill rater_a / rater_b (1-5 groundedness) in this sheet."""
    import pandas as pd
    pd.DataFrame([{"id": r["id"], "answer": r["answer"], "citations": ",".join(r["citations"]),
                   "rater_a": "", "rater_b": ""} for r in recs]).to_csv(path, index=False)


def human_summary(path="human_eval_p1.csv") -> dict:
    df = pd.read_csv(path).dropna(subset=["rater_a", "rater_b"])
    return {"mean_groundedness": round(float((df.rater_a + df.rater_b).mean() / 2), 2),
            "agree_within_1": round(float(((df.rater_a - df.rater_b).abs() <= 1).mean()), 3), "n": len(df)}


def load(path="golden_set.json"):
    return json.load(open(path))
