"""One shared, typed state passed through all agents."""
from dataclasses import dataclass, field
from typing import Any, Optional

ROUTES = ("caption_qa", "style_change", "edit", "domain_generate", "unsupported", "ambiguous")


@dataclass
class State:
    request: str
    image_path: Optional[str] = None
    seed: int = 0
    enable_critic: bool = True
    max_revisions: int = 2
    # filled by agents
    image: Any = None                     # PIL image (validated, RGB)
    route: str = ""
    refused: bool = False
    message: str = ""                     # refusal / clarification / error text shown to the user
    planner: dict = field(default_factory=dict)
    vision: dict = field(default_factory=dict)
    prompt: dict = field(default_factory=dict)      # {prompt, negative_prompt, target_caption}
    generation: dict = field(default_factory=dict)  # {model, prompt, seed, steps, file, nsfw_flag}
    critique: dict = field(default_factory=dict)
    revisions: int = 0
    revision_instruction: str = ""
    output_path: Optional[str] = None
    answer: str = ""                      # text result for caption / question routes
    error: str = ""
    trace: list = field(default_factory=list)
    total_seconds: float = 0.0
