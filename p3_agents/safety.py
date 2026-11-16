"""Rule-based input guardrails (first line of defence; the Planner's LLM check is the second)."""
import re

NSFW = re.compile(r"\b(nude|naked|nsfw|porn\w*|erotic|sexual\w*|topless|lingerie|explicit (sex|content)|genitals?|fetish)\b", re.I)
VIOLENCE = re.compile(r"\b(gore|gory|mutilat\w+|dismember\w*|beheading|torture|child abuse|bloody corpse|decapitat\w+)\b", re.I)
JAILBREAK = re.compile(
    r"(ignore|disregard|forget|override)\b[^.\n]{0,40}\b(instruction|previous|above|prior|rules?|prompt|guidelines?)|"
    r"\b(skip|disable|bypass|turn off|without)\b[^.\n]{0,25}\b(safety|safeguards?|filter|critic|guardrails?|checks?)\b|"
    r"\b(reveal|print|show|output)\b[^.\n]{0,25}\b(system prompt|your instructions|hidden prompt)\b", re.I)
DEFER_TO_IMAGE = re.compile(r"\b(follow|obey|do|execute|carry out)\b[^.\n]{0,30}\b(what|whatever)?\b[^.\n]{0,15}\b(the )?(image|picture|text|note|sign)\b[^.\n]{0,20}\b(says?|instructions?|tells?)\b|"
                            r"\b(instructions?|commands?) (in|on|from) the (image|picture)\b", re.I)
INJECTION_TEXT = re.compile(
    r"(ignore|disregard|forget)\b[^.\n]{0,40}\b(instruction|previous|above|prior|rules?)|"
    r"\b(system prompt|new instructions?|you must|you are now|do not (tell|mention)|override|jailbreak)\b", re.I)


PERSON_WORDS = re.compile(r"\b(she|he|her|him|his|hers|they|them|person|man|woman|girl|boy|people|celebrity|politician|president|"
                          r"actor|actress|face|portrait|photo of)\b", re.I)
HARMFUL_SCENE = re.compile(r"\b(gun|pistol|rifle|weapon|knife|robbery|robber|crime|criminal|arrest\w*|scam|fraud|endorse\w*|"
                           r"bribe\w*|drunk|riot|terroris\w+|kidnap\w*|violent|explosion)\b", re.I)
MAKE_REAL = re.compile(r"\b(look(s)? real|look(s)? authentic|look(s)? genuine|make (it|this|them) (look )?(real|believable)|"
                       r"so (nobody|no one) can tell|pass as real|fake (news|photo|evidence))\b", re.I)


def deceptive_person_edit(request: str) -> bool:
    """Edits that put a person into a harmful scene or make a fabricated claim look real (deceptive fakes)."""
    r = request or ""
    return bool(PERSON_WORDS.search(r) and (HARMFUL_SCENE.search(r) or MAKE_REAL.search(r)))


def text_policy(request: str) -> tuple[bool, str]:
    if NSFW.search(request):
        return True, "Sexual / NSFW content is not allowed."
    if VIOLENCE.search(request):
        return True, "Graphic violence is not allowed."
    if JAILBREAK.search(request):
        return True, "Requests to bypass safety checks or reveal system instructions are not allowed."
    if DEFER_TO_IMAGE.search(request):
        return True, "Instructions hidden in an image are not followed; please state your request in text."
    return False, ""


def image_text_is_injection(visible_text: str) -> bool:
    return bool(INJECTION_TEXT.search(visible_text or ""))


def strip_injection(text: str) -> str:
    return "" if image_text_is_injection(text) else (text or "")
