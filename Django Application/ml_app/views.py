import os
import traceback
from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models.detector2 import DeepfakeDetector
from .models.image_detector import ImageDeepfakeDetector
from .models.audio_detector import AudioDeepfakeDetector
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

video_detector = None
image_detector = None
audio_detector = None

def get_video_detector():
    global video_detector
    if video_detector is None:
        video_detector = DeepfakeDetector()
    return video_detector

def get_image_detector():
    global image_detector
    if image_detector is None:
        image_detector = ImageDeepfakeDetector()
    return image_detector

def get_audio_detector():
    global audio_detector
    if audio_detector is None:
        audio_detector = AudioDeepfakeDetector()
    return audio_detector

@api_view(['POST'])
def analyze_video(request):
    try:
        video_file = request.FILES.get('file')
        if not video_file:
            return Response({'error': 'No video file provided'}, status=400)

        # Create temp directory if it doesn't exist
        temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp')
        os.makedirs(temp_dir, exist_ok=True)

        # Save the uploaded file temporarily
        temp_path = os.path.join(temp_dir, video_file.name)
        print(f"Saving video to temporary path: {temp_path}")
        
        with open(temp_path, 'wb+') as destination:
            for chunk in video_file.chunks():
                destination.write(chunk)

        print(f"Video saved successfully, starting analysis...")
        
        # Analyze the video
        result = get_video_detector().predict(temp_path)
        print(f"Analysis complete: {result}")

        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
            print(f"Temporary file removed: {temp_path}")

        return Response(result)

    except Exception as e:
        print(f"Error in analyze_video: {str(e)}")
        print(traceback.format_exc())
        return Response({
            'error': str(e),
            'detail': traceback.format_exc()
        }, status=500)

@api_view(['POST'])
def analyze_image(request):
    try:
        image_file = request.FILES.get('file')
        if not image_file:
            return Response({'error': 'No image file provided'}, status=400)

        # Create temp directory if it doesn't exist
        temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp')
        os.makedirs(temp_dir, exist_ok=True)

        # Save the uploaded file temporarily
        temp_path = os.path.join(temp_dir, image_file.name)
        print(f"Saving image to temporary path: {temp_path}")
        
        with open(temp_path, 'wb+') as destination:
            for chunk in image_file.chunks():
                destination.write(chunk)

        print(f"Image saved successfully, starting analysis...")
        
        # Analyze the image
        result = get_image_detector().predict(temp_path)
        print(f"Analysis complete: {result}")

        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
            print(f"Temporary file removed: {temp_path}")

        return Response(result)

    except Exception as e:
        print(f"Error in analyze_image: {str(e)}")
        print(traceback.format_exc())
        return Response({
            'error': str(e),
            'detail': traceback.format_exc()
        }, status=500)

@api_view(['POST'])
def analyze_audio(request):
    try:
        audio_file = request.FILES.get('file')
        if not audio_file:
            return Response({'error': 'No audio file provided'}, status=400)

        temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp')
        os.makedirs(temp_dir, exist_ok=True)

        temp_path = os.path.join(temp_dir, audio_file.name)
        print(f"Saving audio to temporary path: {temp_path}")

        with open(temp_path, 'wb+') as destination:
            for chunk in audio_file.chunks():
                destination.write(chunk)

        print(f"Audio saved successfully, starting analysis...")

        result = get_audio_detector().predict(temp_path)
        print(f"Analysis complete: {result}")

        if os.path.exists(temp_path):
            os.remove(temp_path)
            print(f"Temporary file removed: {temp_path}")

        return Response(result)

    except Exception as e:
        print(f"Error in analyze_audio: {str(e)}")
        print(traceback.format_exc())
        return Response({
            'error': str(e),
            'detail': traceback.format_exc()
        }, status=500)