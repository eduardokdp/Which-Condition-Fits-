"""Control every source of randomness used by the pipeline."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)  # inherited by any subprocess
    random.seed(seed)
    np.random.seed(seed)
    try:  # torch is optional; only seeded when an encoder backend is installed
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
