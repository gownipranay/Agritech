import io

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image

from app.schemas.disease import DiseasePredictionResponse
from app.services.disease_inference import get_classifier
from app.services.knowledge_base import get_crop

router = APIRouter()

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024


@router.post("/predict-disease", response_model=DiseasePredictionResponse)
async def predict_disease(file: UploadFile = File(...), crop_id: str | None = Form(default=None)):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported content type: {file.content_type}")

    raw = await file.read()
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image too large (max 10MB)")

    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file")

    classifier = get_classifier()
    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "No trained model checkpoint found. Run ml/disease_classification/train.py "
                "to produce one, or set DISEASE_MODEL_CHECKPOINT."
            ),
        )

    # Scope the prediction to the selected crop's classes when we can, so an
    # out-of-distribution field photo of one crop can't be labelled as another.
    allowed_prefixes = None
    if crop_id:
        crop = get_crop(crop_id)
        if crop and crop.get("vision_classes_prefix"):
            allowed_prefixes = [crop["vision_classes_prefix"]]

    result = classifier.predict(image, allowed_prefixes=allowed_prefixes)
    return DiseasePredictionResponse(**result, model_version="mobilenetv3-small-multicrop-v2")
