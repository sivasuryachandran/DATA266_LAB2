"""Pure-python helpers for Part 1 (no heavy dependencies, unit-tested in tests/)."""
import re

# ---- figure / table labels -------------------------------------------------
_NUM = r"[A-Z]?-?\d+(?:[-–.]\d+)?[a-z]?"
LABEL_RE = re.compile(
    r"\b(?P<kind>(?i:Figure|Fig\.?|Table))\s+(?P<num>" + _NUM + r")"
    r"(?P<corr>\s*\((?i:corrected|revised|updated)\))?")
_CAPTION_VERBS = re.compile(
    r"^\s*(shows?|illustrates?|presents?|displays?|reports?|plots?|compares?|depicts?|"
    r"is|are|was|were|summari[sz]es|gives?|indicates?|suggests?|above|below|and|also)\b", re.I)


def make_label(kind: str, num: str, corrected: bool = False) -> str:
    kind = "Table" if kind.lower().startswith("table") else "Figure"
    num = num.replace("–", "-")
    return f"{kind} {num}" + (" (corrected)" if corrected else "")


def find_labels(text: str) -> list[str]:
    """All figure/table labels mentioned in text, in order, de-duplicated."""
    out = []
    for m in LABEL_RE.finditer(text or ""):
        lab = make_label(m.group("kind"), m.group("num"), bool(m.group("corr")))
        if lab not in out:
            out.append(lab)
    return out


def caption_label(block_text: str) -> str | None:
    """Label if this text block looks like a caption (starts with the label, is short,
    and is not a body sentence such as 'Figure 1-2 shows ...')."""
    t = (block_text or "").strip()
    m = LABEL_RE.match(t)
    if not m or len(t) > 400:
        return None
    rest = t[m.end():]
    if _CAPTION_VERBS.match(rest):
        return None
    return make_label(m.group("kind"), m.group("num"), bool(m.group("corr")))


def base_label(label: str) -> str:
    return label.replace(" (corrected)", "")


# ---- fusion ------------------------------------------------------------------
def rrf(rankings: list[list[str]], k: int = 60, weights: list[float] | None = None) -> list[tuple[str, float]]:
    """Reciprocal rank fusion over ranked id lists."""
    weights = weights or [1.0] * len(rankings)
    score: dict[str, float] = {}
    for w, ranking in zip(weights, rankings):
        for r, i in enumerate(ranking):
            score[i] = score.get(i, 0.0) + w / (k + r + 1)
    return sorted(score.items(), key=lambda kv: -kv[1])


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", (text or "").lower())


# ---- untrusted-content defence ---------------------------------------------
_INJECTION = re.compile(
    r"(ignore|disregard|forget|override)\b[^.\n]{0,40}\b(instruction|previous|above|prior|rules?|prompt)|"
    r"\b(you must|you should now|from now on|new instructions?|system prompt|developer message)\b|"
    r"\b(always|only) (answer|respond|reply|say|output)\b|"
    r"\bdo not (tell|mention|reveal|cite)\b|\bas an ai\b|"
    r"\b(answer|respond|reply) with\b[^.\n]{0,40}\b(only|exactly)\b", re.I)


def sanitize(text: str) -> tuple[str, int]:
    """Drop sentences/lines that read like instructions aimed at the model.
    Returns (clean_text, number_of_sentences_removed)."""
    removed = 0
    lines_out = []
    for line in (text or "").splitlines():
        parts = re.split(r"(?<=[.!?])\s+", line)
        keep = []
        for p in parts:
            if _INJECTION.search(p):
                removed += 1
            else:
                keep.append(p)
        lines_out.append(" ".join(keep))
    return "\n".join(lines_out), removed


def has_injection(text: str) -> bool:
    return bool(_INJECTION.search(text or ""))


# ---- conversation cues (generic, not question specific) ----------------------
_OVERRIDE = re.compile(
    r"\b(assume|suppose|let'?s say|imagine|pretend|hypothetically|for this (chat|conversation|session)|"
    r"for now|in this (chat|conversation)|what if|if we (take|use|set))\b", re.I)
_PERMANENT = re.compile(
    r"\b(permanent(ly)?|update|change|edit|modify|correct|overwrite|replace|fix)\b[^.?\n]{0,60}"
    r"\b(document|documents|knowledge base|database|source|pdf|file|record)s?\b|"
    r"\b(save|store|remember)\b[^.?\n]{0,40}\b(permanent(ly)?|for (all|every) (future )?(user|conversation|session)s?)\b|"
    r"\bfrom now on\b[^.?\n]{0,40}\b(document|documents|knowledge base)\b", re.I)


def looks_like_override(q: str) -> bool:
    return bool(_OVERRIDE.search(q or ""))


def looks_like_permanent_change(q: str) -> bool:
    return bool(_PERMANENT.search(q or ""))


# ---- output formatting ---------------------------------------------------------
def clip_words(text: str, n: int = 120) -> str:
    words = (text or "").split()
    return " ".join(words[:n])


_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


def numbers_in(text: str) -> list[float]:
    out = []
    for m in _NUMBER.findall(text or ""):
        try:
            out.append(float(m.replace(",", "").rstrip(".")))
        except ValueError:
            pass
    return out


def numbers_match(gold: list[float], text: str, tol: float = 0.01) -> bool:
    """Every gold number appears in text within +/- tol (relative; absolute for ~0)."""
    found = numbers_in(text)
    for g in gold:
        if not any(abs(f - g) <= tol * abs(g) + (1e-9 if g else 1e-6) for f in found):
            return False
    return True
