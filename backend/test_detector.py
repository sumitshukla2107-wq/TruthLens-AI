
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models" / "convnext" / "AI Images Detector"

sys.path.insert(0, str(MODEL_DIR))

from model.convnext import build_model
from transforms import test_transforms

CHECKPOINT = MODEL_DIR / "checkpoints" / "checkpoint_phase2.pth"

def main():
    print("Loading model on CPU...")

    model = build_model()

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(checkpoint["model"])
    model.eval()

    print("Model loaded successfully!")

    image_path = input("Enter the full path of a test image: ").strip().strip('"')

    with Image.open(image_path) as img:
        image = img.convert("RGB")

    tensor = test_transforms()(image).unsqueeze(0)

    with torch.inference_mode():
        probabilities = F.softmax(model(tensor), dim=1)[0]

    real_prob = probabilities[0].item()
    fake_prob = probabilities[1].item()

    print(f"Real probability: {real_prob:.2%}")
    print(f"AI-generated probability: {fake_prob:.2%}")
    print("Prediction:", "AI-generated" if fake_prob > 0.5 else "Real")
    print("Note: This is a model estimate, not proof of authenticity.")

if __name__ == "__main__":
    main()
