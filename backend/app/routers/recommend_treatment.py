from fastapi import APIRouter, HTTPException

router = APIRouter()


@router.post("/recommend-treatment")
async def recommend_treatment():
    """Phase 4 — not yet implemented.

    Will be a strict retrieval pipeline: disease -> active ingredient
    (curated, cited lookup table) -> matching products (retrieval, never
    LLM-generated brand names). See README "Why this system doesn't let
    the LLM freely recommend chemicals" for the design this will follow.
    """
    raise HTTPException(
        status_code=501, detail="recommend-treatment is planned for Phase 4, not yet implemented"
    )
