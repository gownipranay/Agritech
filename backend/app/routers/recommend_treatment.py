from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.knowledge_base import get_treatment, global_disclaimer

router = APIRouter()


class TreatmentRequest(BaseModel):
    disease_key: str


@router.post("/recommend-treatment")
async def recommend_treatment(req: TreatmentRequest):
    """Grounded disease -> management lookup (Phase 4).

    This is a deterministic dictionary read from the curated, cited
    knowledge base — never an LLM generation. A brand/active ingredient can
    only ever be returned because its disease key matched an entry that was
    seeded from ICAR / State Agricultural University advisories. See the
    README section on why the LLM is never allowed to invent chemicals.
    """
    entry = get_treatment(req.disease_key)
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail=f"No treatment entry for '{req.disease_key}'. It may be a healthy class or unknown.",
        )
    return {
        "disease_key": req.disease_key,
        **entry,
        "disclaimer": global_disclaimer(),
    }
