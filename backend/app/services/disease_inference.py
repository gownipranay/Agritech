"""Loads the trained disease classifier once and exposes it to the API layer.

The ml/ package isn't pip-installed (no setup.py/pyproject yet — prototype
stage), so we add it to sys.path at import time rather than duplicating the
model/dataset code inside backend/.
"""

import sys
from functools import lru_cache
from pathlib import Path

from app.core.config import settings

ML_DIR = Path(__file__).resolve().parents[3] / "ml" / "disease_classification"
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from infer import DiseaseClassifier  # noqa: E402


@lru_cache(maxsize=1)
def get_classifier() -> DiseaseClassifier | None:
    checkpoint = Path(settings.disease_model_checkpoint)
    if not checkpoint.exists():
        return None
    return DiseaseClassifier(str(checkpoint))
