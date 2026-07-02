from fastapi import APIRouter, HTTPException

from app.services.knowledge_base import (
    fertilizer_disclaimer,
    get_crop,
    get_crops,
    get_fertilizer,
)

router = APIRouter()


@router.get("/crops")
async def list_crops():
    """Catalogue of crops the app supports, with their vision-detection status.

    vision_support is one of:
      - vision_supported: the trained CV model classifies photos of this crop
      - vision_proxy: photos are routed to a closely-related trained crop
        (chilli -> bell pepper) with a caveat
      - advisory_only: no trained detector yet; symptom-based guidance only
    """
    return {"crops": get_crops()}


@router.get("/crops/{crop_id}")
async def crop_detail(crop_id: str):
    crop = get_crop(crop_id)
    if crop is None:
        raise HTTPException(status_code=404, detail=f"Unknown crop: {crop_id}")
    return crop


@router.get("/crops/{crop_id}/fertilizer")
async def crop_fertilizer(crop_id: str):
    """Per-crop fertilizer / nutrient recommendation (NPK rate, split schedule,
    branded fertilizer products). Deterministic lookup from the curated
    knowledge base — soil-test-based dosing, not an LLM guess.
    """
    fert = get_fertilizer(crop_id)
    if fert is None:
        raise HTTPException(status_code=404, detail=f"No fertilizer data for crop: {crop_id}")
    return {"crop_id": crop_id, **fert, "disclaimer": fertilizer_disclaimer()}
