from fastapi import APIRouter, HTTPException

router = APIRouter()


@router.post("/predict-yield")
async def predict_yield():
    """Phase 3 — not yet implemented.

    Will accept crop type, soil NPK, rainfall, region, and past disease
    incidence, and return a yield estimate with a confidence interval
    (not just a point prediction).
    """
    raise HTTPException(status_code=501, detail="predict-yield is planned for Phase 3, not yet implemented")
