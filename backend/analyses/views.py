import os
import tempfile
import cv2
from .detector import predict_image
from .video_detector import predict_video
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status



@api_view(["GET"])
def health_check(request):
    return Response({
        "status": "online",
        "project": "TruthLens AI",
        "message": "Backend is working successfully!",
        "model_status": "not_configured"
    })

from io import BytesIO

from PIL import Image, UnidentifiedImageError
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status


MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
def analyze_image(request):
    uploaded_file = request.FILES.get("image")

    if not uploaded_file:
        return Response(
            {"error": "Please upload an image."},
            status=status.HTTP_400_BAD_REQUEST
        )

    if uploaded_file.size > MAX_FILE_SIZE:
        return Response(
            {"error": "Maximum image size is 10 MB."},
            status=status.HTTP_400_BAD_REQUEST
        )

    try:
        image_bytes = uploaded_file.read()

        with Image.open(BytesIO(image_bytes)) as image:
            image_format = image.format

            if image_format not in ALLOWED_FORMATS:
                return Response(
                    {"error": "Only JPEG, PNG and WebP are supported."},
                    status=status.HTTP_400_BAD_REQUEST
                )

            image.verify()
    
        # Reopen the validated image and run the AI detector.
        with Image.open(BytesIO(image_bytes)) as image:
            result = predict_image(image)

        return Response({
            "project": "TruthLens AI",
            "status": "analyzed",
            "filename": uploaded_file.name,
            "format": image_format,
            "model_status": "ready",
            "prediction": result["label"],
            "real_probability": result["real_probability"],
            "ai_probability": result["ai_probability"],
            "message": (
                "Image analyzed successfully. Results are "
                "model estimates, not proof of authenticity."
            )
        })

        

    except (UnidentifiedImageError, OSError, ValueError):
        return Response(
            {"error": "The uploaded file is not a valid supported image."},
            status=status.HTTP_400_BAD_REQUEST
        )

MAX_VIDEO_SIZE = 250 * 1024 * 1024
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi"}


@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
def analyze_video(request):
    uploaded_file = request.FILES.get("video")

    if not uploaded_file:
        return Response(
            {"error": "Please upload a video."},
            status=status.HTTP_400_BAD_REQUEST
        )

    if uploaded_file.size > MAX_VIDEO_SIZE:
        return Response(
            {"error": "Maximum video size is 250 MB."},
            status=status.HTTP_400_BAD_REQUEST
        )

    extension = os.path.splitext(uploaded_file.name)[1].lower()

    if extension not in ALLOWED_VIDEO_EXTENSIONS:
        return Response(
            {"error": "Only MP4, MOV and AVI videos are supported."},
            status=status.HTTP_400_BAD_REQUEST
        )

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=extension, delete=False
        ) as temp_file:
            temp_path = temp_file.name

            for chunk in uploaded_file.chunks():
                temp_file.write(chunk)

        video = cv2.VideoCapture(temp_path)

        try:
            if not video.isOpened():
                return Response(
                    {"error": "The uploaded file is not a readable video."},
                    status=status.HTTP_400_BAD_REQUEST
                )

            success, frame = video.read()

            if not success or frame is None:
                return Response(
                    {"error": "Could not read a frame from this video."},
                    status=status.HTTP_400_BAD_REQUEST
                )

            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = video.get(cv2.CAP_PROP_FPS)

        finally:
            video.release()

        
        result = predict_video(temp_path)

        return Response({
            "project": "TruthLens AI",
            "status": "analyzed",
            "filename": uploaded_file.name,
            "size_mb": round(
                uploaded_file.size / (1024 * 1024), 2
            ),
            "frame_count": frame_count,
            "fps": round(fps, 2) if fps > 0 else None,
            "model_status": "ready",
            "prediction": result["prediction"],
            "real_probability": result["real_probability"],
            "fake_probability": result["fake_probability"],
            "frames_analyzed": result["frames_analyzed"],
            "message": (
                "Video analysis complete. Predictions are model "
                "estimates, not proof of authenticity."
            ),
        })

    except (OSError, ValueError, cv2.error):
        return Response(
            {"error": "Could not process this video file."},
            status=status.HTTP_400_BAD_REQUEST
        )

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass

