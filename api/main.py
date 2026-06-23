import io
import time
import torch

from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from torchvision import models

from src.transforms import val_transform


# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "models/resnet18_best.pth"

CLASS_NAMES = ["early_blight", "healthy", "late_blight", "leaf_mold"]


# -----------------------------
# RESPONSE SCHEMA
# -----------------------------
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class PredictionResponse(BaseModel):
    class_name: str
    confidence: float
    inference_ms: float


# -----------------------------
# FASTAPI APP
# -----------------------------
app = FastAPI(
    title="Leaf Disease Detection API",
    version="1.0.0",
    description="ResNet18 Leaf Disease Classifier"
)


# -----------------------------
# LOAD MODEL
# -----------------------------
@app.on_event("startup")
async def load_model():

    model = models.resnet18(weights=None)

    model.fc = torch.nn.Linear(
        model.fc.in_features,
        len(CLASS_NAMES)
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location="cpu"
        )
    )

    model.eval()

    app.state.model = model


# -----------------------------
# HEALTH CHECK
# -----------------------------
@app.get("/health", response_model=HealthResponse)
async def health():

    return HealthResponse(
        status="ok",
        model_loaded=app.state.model is not None
    )


# -----------------------------
# PREDICTION
# -----------------------------
@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...)
):

    if file.content_type not in {
        "image/jpeg",
        "image/png"
    }:
        raise HTTPException(
            status_code=415,
            detail="Unsupported media type"
        )

    raw = await file.read()

    try:
        image = Image.open(
            io.BytesIO(raw)
        ).convert("RGB")

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file"
        )

    tensor = val_transform(image)
    tensor = tensor.unsqueeze(0)

    start = time.perf_counter()

    with torch.no_grad():

        logits = app.state.model(tensor)

        probs = torch.softmax(
            logits,
            dim=1
        )

        confidence, pred_idx = torch.max(
            probs,
            dim=1
        )

    inference_ms = (
        time.perf_counter() - start
    ) * 1000

    return PredictionResponse(
        class_name=CLASS_NAMES[pred_idx.item()],
        confidence=round(
            confidence.item(),
            4
        ),
        inference_ms=round(
            inference_ms,
            2
        )
    )