import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

class ImageDeepfakeDetector:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), 'cnn_model.h5')
        print("Initializing Image Deepfake Detector...")
        self.target_size = (128, 128)  # Changed to match video model input size
        
        try:
            # Load the same CNN model used for video detection
            self.model = load_model(model_path)
            print("Model loaded successfully")
            
            # Warm up the model with correct input shape
            dummy_input = np.zeros((1, 128, 128, 3))  # Changed dimensions
            self.model.predict(dummy_input)
            print("Model warm-up complete")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def preprocess_image(self, image):
        try:
            # Convert to RGB if needed
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            elif image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Resize to match model's expected input size
            image = cv2.resize(image, self.target_size)
            
            # Normalize pixel values to [0, 1]
            image = image.astype(np.float32) / 255.0
            
            # Add batch dimension
            image = np.expand_dims(image, axis=0)
            
            return image
            
        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            raise

    def predict(self, image_path):
        try:
            print(f"Starting image analysis for {image_path}")
            
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not read image file")
            
            # Get original image details
            height, width = image.shape[:2]
            
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Make prediction
            prediction = self.model.predict(processed_image)[0][0]
            
            # Convert prediction to result
            is_fake = prediction > 0.5
            confidence = float(prediction * 100) if is_fake else float((1 - prediction) * 100)
            
            result = {
                'result': 'FAKE' if is_fake else 'REAL',
                'confidence': confidence,
                'image_size': [height, width, image.shape[2]],
                'input_shape': self.target_size + (3,)
            }
            
            print(f"Image analysis complete. Result: {result['result']} with {result['confidence']:.2f}% confidence")
            return result

        except Exception as e:
            print(f"Error analyzing image: {str(e)}")
            raise