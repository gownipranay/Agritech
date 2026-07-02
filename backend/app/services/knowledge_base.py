"""Loads the curated crop + treatment knowledge base once.

These JSON files are the deterministic source of truth for Phase 4. The
treatment lookup is a plain dictionary read — never a model call — so an LLM
summarizer downstream cannot introduce a chemical name that isn't already
here. See the project README, "Why this system doesn't let the LLM freely
recommend chemicals".
"""

import json
from functools import lru_cache
from pathlib import Path

from app.core.config import settings

KB_DIR = Path(settings.knowledge_base_dir)


@lru_cache(maxsize=1)
def _load(name: str) -> dict:
    path = KB_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Knowledge base file missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def get_crops() -> list[dict]:
    return _load("crops.json")["crops"]


def get_crop(crop_id: str) -> dict | None:
    return next((c for c in get_crops() if c["id"] == crop_id), None)


def get_treatments() -> dict:
    return _load("treatments.json")["treatments"]


def get_treatment(disease_key: str) -> dict | None:
    return get_treatments().get(disease_key)


def global_disclaimer() -> str:
    return _load("treatments.json")["_disclaimer"]


def products_disclaimer() -> str:
    return _load("products.json")["_disclaimer"]


def match_products(active_ingredients: list[str]) -> list[dict]:
    """Retrieval, not generation: return branded products whose match_keywords
    are contained in one of the already-selected active ingredients. A brand
    can only surface because its active ingredient matched the diagnosis.
    """
    haystacks = [a.lower() for a in active_ingredients]
    matched = []
    for prod in _load("products.json")["products"]:
        for kw in prod["match_keywords"]:
            if any(kw in h for h in haystacks):
                matched.append({k: v for k, v in prod.items() if k != "match_keywords"})
                break
    return matched
