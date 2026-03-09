from django.urls import path
from . import views

urlpatterns = [
    path('api/analyze/', views.analyze_video, name='analyze_video'),
    path('api/analyze-image/', views.analyze_image, name='analyze_image'),
    path('api/analyze-audio/', views.analyze_audio, name='analyze_audio'),
]