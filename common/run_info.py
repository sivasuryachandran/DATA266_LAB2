"""Seeding and the run-information cell required by C1.4."""
import json
import random
import subprocess
import sys

SEED = 266


def seed_everything(seed: int = SEED):
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def run_info(seed: int = SEED) -> dict:
    import torch
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    return {"commit": commit, "python": sys.version.split()[0], "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "seed": seed}


def print_run_info(seed: int = SEED):
    seed_everything(seed)
    print(json.dumps(run_info(seed)))
