"""Orchestration in plain Python: Planner -> Vision -> Prompt Eng -> Generation -> Critique -> accept | revise (max 2)."""
import json
import os
import time

from p3_agents import agents
from p3_agents.state import State

TRACE_DIR = "work_p3/traces"


def load_image(path):
    """Returns (PIL image or None, error message)."""
    from PIL import Image
    import numpy as np
    if not path:
        return None, ""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None, "The uploaded file is missing or empty."
    try:
        img = Image.open(path)
        img.load()
        img = img.convert("RGB")
    except Exception:
        return None, "The file is not a readable image (it may be corrupted)."
    if min(img.size) < 32:
        return None, f"The image is too small ({img.size[0]}x{img.size[1]}); please upload at least 64x64 pixels."
    if float(np.asarray(img).std()) < 2.0:
        return None, "The image looks blank or uniform; please upload a clearer image."
    if max(img.size) > 1024:
        img.thumbnail((1024, 1024))
    return img, ""


def _step(s: State, name, fn, hub, summary):
    t0 = time.time()
    entry = {"agent": name, "input": summary(s), "output": None, "seconds": None}
    s.trace.append(entry)
    try:
        fn(s, hub)
    except Exception as e:                    # never crash: turn it into a message
        s.error = f"{name} failed: {type(e).__name__}: {str(e)[:200]}"
        s.message = "Sorry, something went wrong while processing your request. " + (
            "The GPU ran out of memory; try a smaller image." if "out of memory" in str(e).lower() else "Please try again or rephrase.")
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    entry["seconds"] = round(time.time() - t0, 2)
    entry["output"] = {"planner": s.planner, "vision": s.vision, "prompt_engineer": s.prompt, "generator": s.generation,
                       "critic": s.critique}.get({"Planner": "planner", "Vision": "vision", "PromptEngineer": "prompt_engineer",
                                                  "ImageGeneration": "generator", "Critique": "critic"}[name]) or {"message": s.message}
    return not s.error


def run(hub, request, image_path=None, seed=0, enable_critic=True, max_revisions=2, trace_dir=TRACE_DIR) -> State:
    t0 = time.time()
    s = State(request=request or "", image_path=image_path, seed=seed, enable_critic=enable_critic, max_revisions=min(max_revisions, 2))
    s.image, err = load_image(image_path)
    if err:
        s.message, s.error = err, "input_image_invalid"
    else:
        flow = _run_flow(s, hub)
    s.total_seconds = round(time.time() - t0, 2)
    _save_trace(s, trace_dir)
    return s


def _run_flow(s, hub):
    if not _step(s, "Planner", agents.planner, hub, lambda s: {"request": s.request, "has_image": s.image is not None}):
        return
    if s.refused or s.route in ("ambiguous", "unsupported"):
        return
    if s.route in ("caption_qa", "style_change", "edit"):
        if not _step(s, "Vision", agents.vision, hub, lambda s: {"route": s.route}):
            return
        if s.route == "caption_qa":
            return
    while True:
        if not _step(s, "PromptEngineer", agents.prompt_engineer, hub,
                     lambda s: {"route": s.route, "revision_instruction": s.revision_instruction, "revision": s.revisions}):
            return
        if not _step(s, "ImageGeneration", agents.generate, hub, lambda s: {"prompt": s.prompt, "revision": s.revisions}):
            return
        if not s.enable_critic:
            return
        if not _step(s, "Critique", agents.critic, hub, lambda s: {"prompt": s.prompt.get("target_caption")}):
            return
        d = s.critique["decision"]
        if d == "accept":
            return
        if d == "reject_unsafe":
            s.output_path, s.refused = None, True
            s.message = "The generated image was withheld by the output safety check."
            return
        if s.revisions >= s.max_revisions:
            s.message = f"Returned after {s.revisions} revision(s); the critic still rates the result as imperfect: {s.revision_instruction}"
            return
        s.revisions += 1


def _save_trace(s: State, trace_dir):
    os.makedirs(trace_dir, exist_ok=True)
    rec = {"request": s.request, "image": s.image_path, "route": s.route, "refused": s.refused, "message": s.message,
           "error": s.error, "answer": s.answer, "output": s.output_path, "revisions": s.revisions,
           "critic_enabled": s.enable_critic, "total_seconds": s.total_seconds, "steps": s.trace}
    name = f"{int(time.time() * 1000)}.json"
    json.dump(rec, open(os.path.join(trace_dir, name), "w"), indent=1, default=str)
