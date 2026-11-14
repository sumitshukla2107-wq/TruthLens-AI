
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import (
    VideoMAEForVideoClassification,
    VideoMAEImageProcessor,
)

MODEL_ID = "Vansh180/VideoMae-ffc23-deepfake-detector"

VIDEO_PATH = "test_video_2.mp4"  # Apni video ka naam yahan likho

print("Loading model...")
processor = VideoMAEImageProcessor.from_pretrained(MODEL_ID)
model = VideoMAEForVideoClassification.from_pretrained(MODEL_ID)
model.eval()

print("Reading video...")
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise SystemExit(f"Could not open video: {VIDEO_PATH}")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if total_frames < 1:
    cap.release()
    raise SystemExit("Video contains no readable frames.")

# Video se 16 evenly spaced frames select karo
frame_indices = np.linspace(
    0, total_frames - 1, 16, dtype=int
)

frames = []

for index in frame_indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    success, frame = cap.read()

    if success:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))

cap.release()

if len(frames) != 16:
    raise SystemExit(
        f"Could read only {len(frames)} frames; expected 16."
    )

from pathlib import Path

debug_dir = Path("debug_frames")
debug_dir.mkdir(exist_ok=True)

for i, frame in enumerate(frames, start=1):
    frame.save(debug_dir / f"frame_{i:02d}.jpg")

print("Saved 16 debug frames to:", debug_dir.resolve())

print("Processing 16 frames...")

print("\nChecking individual frames...")

for i, frame in enumerate(frames, start=1):
    single_inputs = processor([frame] * 16, return_tensors="pt")

    with torch.inference_mode():
        single_outputs = model(**single_inputs)
        single_probs = torch.softmax(
            single_outputs.logits, dim=-1
        )[0]

    real_pct = single_probs[0].item() * 100
    fake_pct = single_probs[1].item() * 100

    print(
        f"Frame {i:02d}: "
        f"Real={real_pct:.2f}% | Fake={fake_pct:.2f}%"
    )


inputs = processor(frames, return_tensors="pt")
print("Input tensor shape:", inputs["pixel_values"].shape)
print("Unique sampled frames:", len({
    frame.tobytes() for frame in frames
}))
print("Model labels:", model.config.id2label)

with torch.inference_mode():
    outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=-1)[0]
print("Raw logits:", outputs.logits)
print("Softmax probabilities:", probabilities)
print("Predicted class index:", probabilities.argmax().item())
print("Predicted class label:", model.config.id2label[
    probabilities.argmax().item()
])
real_probability = probabilities[0].item() * 100
fake_probability = probabilities[1].item() * 100

prediction = (
    "Fake" if fake_probability > real_probability else "Real"
)

print("\n--- TruthLens AI Video Result ---")
print("Prediction:", prediction)
print(f"Real probability: {real_probability:.2f}%")
print(f"Fake probability: {fake_probability:.2f}%")
print("Frames analyzed:", len(frames))
print("\nNote: Model predictions are estimates, not proof.")
