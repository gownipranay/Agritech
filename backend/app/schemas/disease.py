from pydantic import BaseModel


class ClassConfidence(BaseModel):
    class_name: str
    confidence: float


class DiseasePredictionResponse(BaseModel):
    predicted_class: str | None
    confidence: float
    is_confident: bool
    message: str | None
    top_5: list[dict]
    model_version: str
