"""2.3 Fixed prompt set: 10 in-domain (test captions, seeded), 5 out-of-domain, 5 compositional.
Written to prompts.json BEFORE training finishes and committed."""
import json
import random

OUT_OF_DOMAIN = [
    "a photograph of a red sports car on a coastal highway at sunset",
    "a bowl of fresh fruit on a wooden table, studio lighting",
    "a snowy mountain village at night, realistic photo",
    "a golden retriever puppy sitting on a green lawn",
    "a modern glass skyscraper reflecting the clouds",
]
COMPOSITIONAL = [
    "two cartoon characters, one in a red outfit and one in a blue outfit, standing side by side",
    "a cartoon character holding a green umbrella while a smaller character sits on a yellow chair",
    "three characters in a row: the left one in orange, the middle one in purple, the right one in white",
    "a character with blue hair next to a character with red hair, in front of a large tree",
    "a character on the left looking at a character on the right, who is holding a sword",
]


def make_prompts(test_captions, path="prompts.json", seed=266):
    rng = random.Random(seed)
    in_domain = rng.sample(sorted(set(test_captions)), 10)
    prompts = ([{"id": f"in{i:02d}", "type": "in_domain", "prompt": p} for i, p in enumerate(in_domain)] +
               [{"id": f"ood{i:02d}", "type": "out_of_domain", "prompt": p} for i, p in enumerate(OUT_OF_DOMAIN)] +
               [{"id": f"comp{i:02d}", "type": "compositional", "prompt": p} for i, p in enumerate(COMPOSITIONAL)])
    json.dump(prompts, open(path, "w"), indent=1)
    return prompts
