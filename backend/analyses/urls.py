from django.urls import path
from .views import health_check

urlpatterns = [
    path("health/", health_check, name="health"),
]

from django.urls import path
from .views import health_check, analyze_image

urlpatterns = [
    path("health/", health_check, name="health"),
    path("analyze/image/", analyze_image, name="analyze-image"),
]
from django.urls import path
from .views import health_check, analyze_image, analyze_video

urlpatterns = [
    path("health/", health_check, name="health"),
    path("analyze/image/", analyze_image, name="analyze-image"),
    path("analyze/video/", analyze_video, name="analyze-video"),
]