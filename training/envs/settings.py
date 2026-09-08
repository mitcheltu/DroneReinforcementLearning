from functools import lru_cache
from pathlib import Path

from training.contracts import load_config_bundle

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=8)
def settings(shared=None):
    return load_config_bundle(Path(shared) if shared else ROOT / "shared")
