import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import cv2
from mtcnn import MTCNN
from pathlib import Path

class DeepfakeDetector:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), 'cnn_model.h5')
        print("Initializing Deepfake Detector...")
        self.model_path = model_path
        self.model = self.load_model()
        self.detector = MTCNN()
        self.target_size = (128, 128)

    def load_model(self):
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found at {self.model_path}")

            print(f"Loading model from {self.model_path}")
            model = load_model(self.model_path, compile=False)
            print("Model loaded successfully")
            return model
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def detect_and_crop_face(self, frame):
        try:
            # Convert BGR to RGB for MTCNN
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.detector.detect_faces(frame_rgb)
            
            if results:
                bounding_box = results[0]['box']
                x, y, width, height = bounding_box
                # Add padding to the bounding box
                padding = int(min(width, height) * 0.1)
                x = max(0, x - padding)
                y = max(0, y - padding)
                width = min(frame.shape[1] - x, width + 2*padding)
                height = min(frame.shape[0] - y, height + 2*padding)
                
                face = frame[y:y+height, x:x+width]
                face = cv2.resize(face, self.target_size)
                return face, True
            else:
                # If no face is detected, return the resized full frame
                return cv2.resize(frame, self.target_size), False
                
        except Exception as e:
            print(f"Error detecting face: {str(e)}")
            return cv2.resize(frame, self.target_size), False

    def preprocess_face(self, face):
        try:
            # Convert BGR to RGB
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            # Convert to array and normalize
            img_array = image.img_to_array(face_rgb)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0  # Normalize to [0, 1]
            return img_array
            
        except Exception as e:
            print(f"Error preprocessing face: {str(e)}")
            raise

    def predict_frame(self, frame):
        try:
            # Detect and crop face
            face, face_detected = self.detect_and_crop_face(frame)
            
            # Preprocess the face
            processed_face = self.preprocess_face(face)
            
            # Get prediction
            prediction = self.model.predict(processed_face)
            
            # Handle different model output formats
            if isinstance(prediction, list):
                prediction = prediction[0]
            elif len(prediction.shape) > 1 and prediction.shape[1] > 1:
                # If model outputs probabilities for both classes
                prediction = prediction[0][1]  # Take fake probability
            else:
                prediction = prediction[0][0]  # Single output
            
            # Get result
            is_fake = prediction >= 0.5  # Adjust threshold if needed
            confidence = float(prediction if is_fake else 1 - prediction) * 100
            
            return is_fake, confidence, face_detected
            
        except Exception as e:
            print(f"Error predicting frame: {str(e)}")
            raise

    def extract_frames(self, video_path, max_frames=32):
        print(f"Extracting frames from {video_path}")
        frames = []
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_indices = np.linspace(0, total_frames-1, max_frames, dtype=int)
            
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
            cap.release()
        except Exception as e:
            print(f"Error extracting frames: {str(e)}")
            raise

        return frames

    def predict(self, video_path):
        try:
            print(f"Starting video analysis for {video_path}")
            
            # Extract frames
            frames = self.extract_frames(video_path)
            if len(frames) == 0:
                raise ValueError("No frames could be extracted from the video")

            return self.analyze_video(frames)

        except Exception as e:
            print(f"Error analyzing video: {str(e)}")
            raise

    def analyze_video(self, frames):
        try:
            predictions = []
            confidences = []
            faces_detected = []
            processed_frames = []

            for i, frame in enumerate(frames):
                print(f"Analyzing frame {i+1}/{len(frames)}")
                is_fake, confidence, face_detected = self.predict_frame(frame)
                
                predictions.append(is_fake)
                confidences.append(confidence)
                faces_detected.append(face_detected)
                
                # Only consider frames where a face was detected
                if face_detected:
                    processed_frames.append(frame)

            # Calculate results only from frames with detected faces
            if any(faces_detected):
                valid_predictions = [p for p, d in zip(predictions, faces_detected) if d]
                valid_confidences = [c for c, d in zip(confidences, faces_detected) if d]
                
                fake_ratio = sum(valid_predictions) / len(valid_predictions)
                avg_confidence = sum(valid_confidences) / len(valid_confidences)
            else:
                fake_ratio = 0
                avg_confidence = 0

            # Final decision based on majority voting
            is_fake = fake_ratio > 0.5
            result = {
                'result': 'FAKE' if is_fake else 'REAL',
                'confidence': avg_confidence,
                'frame_predictions': list(zip(predictions, confidences)),
                'faces_detected': faces_detected,
                'total_frames': len(frames),
                'frames_with_faces': sum(faces_detected)
            }
            
            print(f"Analysis complete. Result: {result['result']} with {result['confidence']:.2f}% confidence")
            print(f"Faces detected in {result['frames_with_faces']}/{result['total_frames']} frames")
            
            return result
            
        except Exception as e:
            print(f"Error analyzing video: {str(e)}")
            raise