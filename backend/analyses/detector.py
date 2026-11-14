
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models" / "convnext" / "AI Images Detector"
CHECKPOINT_PATH = (
    MODEL_DIR / "checkpoints" / "checkpoint_phase2.pth"
)

# Add the downloaded model folder to Python's import path.
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from model.convnext import build_model
from transforms import test_transforms

_model = None


def get_model():
    global _model

    if _model is None:
        if not CHECKPOINT_PATH.is_file():
            raise FileNotFoundError(
                f"Model checkpoint not found: {CHECKPOINT_PATH}"
            )

        model = build_model()

        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location="cpu",
            weights_only=True,
        )

        model.load_state_dict(checkpoint["model"])
        model.eval()
        _model = model

    return _model


def predict_image(image: Image.Image):
    model = get_model()

    image = image.convert("RGB")
    tensor = test_transforms()(image).unsqueeze(0)

    with torch.inference_mode():
        logits = model(tensor)
        probabilities = F.softmax(logits, dim=1)[0]

    # Class 0 = Real, Class 1 = AI-generated,
    # matching the existing successful test script.
    real_probability = probabilities[0].item()
    ai_probability = probabilities[1].item()

    label = (
        "AI-generated"
        if ai_probability > real_probability
        else "Real"
    )

    return {
        "label": label,
        "real_probability": round(real_probability * 100, 2),
        "ai_probability": round(ai_probability * 100, 2),
    }
