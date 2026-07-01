from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.routers import advisory, predict_disease, predict_yield, recommend_treatment

app = FastAPI(
    title="AgriAI",
    description="Multimodal crop health & yield advisory API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_disease.router, tags=["disease"])
app.include_router(predict_yield.router, tags=["yield"])
app.include_router(recommend_treatment.router, tags=["treatment"])
app.include_router(advisory.router, tags=["advisory"])


@app.get("/health")
async def health():
    return {"status": "ok"}
