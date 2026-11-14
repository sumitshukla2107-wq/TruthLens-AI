
import torch
from transformers import (
    VideoMAEForVideoClassification,
    VideoMAEImageProcessor,
)

MODEL_ID = "Vansh180/VideoMae-ffc23-deepfake-detector"

print("Loading VideoMAE processor...")

processor = VideoMAEImageProcessor.from_pretrained(MODEL_ID)

print("Loading VideoMAE model...")

model = VideoMAEForVideoClassification.from_pretrained(MODEL_ID)
model.eval()

print("\nModel loaded successfully!")
print("PyTorch:", torch.__version__)
print("Device: CPU")
print("Labels:", model.config.id2label)
print("Expected frames:", model.config.num_frames)
print("Input size:", processor.size)
