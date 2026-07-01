from fastapi import APIRouter, HTTPException

router = APIRouter()


@router.post("/advisory")
async def advisory():
    """Phase 4/5 — not yet implemented.

    Will call the Claude API to turn the structured outputs of
    /predict-disease, /predict-yield, and /recommend-treatment into a
    farmer-readable, localized summary. The LLM will only be allowed to
    summarize/translate structured input, never invent chemical names,
    brands, or dosages.
    """
    raise HTTPException(status_code=501, detail="advisory is planned for Phase 4/5, not yet implemented")
