
import cv2
import torch
from PIL import Image
from transformers import (
    VideoMAEForVideoClassification,
    VideoMAEImageProcessor,
)

MODEL_ID = "Vansh180/VideoMae-ffc23-deepfake-detector"

_processor = None
_model = None


def load_video_model():
    global _processor, _model

    if _model is None:
        _processor = VideoMAEImageProcessor.from_pretrained(MODEL_ID)
        _model = VideoMAEForVideoClassification.from_pretrained(MODEL_ID)
        _model.eval()

    return _processor, _model


def predict_video(video_path):
    processor, model = load_video_model()

    video = cv2.VideoCapture(video_path)

    try:
        if not video.isOpened():
            raise ValueError("Could not open the uploaded video.")

        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames < 1:
            raise ValueError("The video contains no readable frames.")

        # Select 16 frames distributed across the video.
        frame_indices = [
            int(i * (total_frames - 1) / 15)
            for i in range(16)
        ]

        frames = []

        for frame_index in frame_indices:
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = video.read()

            if not success or frame is None:
                raise ValueError(
                    f"Could not read video frame {frame_index}."
                )

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))

    finally:
        video.release()

    inputs = processor(frames, return_tensors="pt")

    with torch.inference_mode():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)[0]

    real_probability = float(probabilities[0].item())
    fake_probability = float(probabilities[1].item())

    prediction = (
        "Real"
        if real_probability >= fake_probability
        else "Fake"
    )

    return {
        "prediction": prediction,
        "real_probability": round(real_probability * 100, 2),
        "fake_probability": round(fake_probability * 100, 2),
        "frames_analyzed": len(frames),
    }
