import os
import librosa
import numpy as np
import soundfile as sf
from tensorflow.keras.models import load_model

class AudioDeepfakeDetector:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), 'my_model.h5')
        print("Initializing Audio Deepfake Detector...")
        try:
            self.model = self.load_model(model_path)
            self.target_sr = 16000
            self.n_mfcc = 40  # Changed to match original code
            self.max_length = 500  # Changed to match original code
            print("Audio detector initialized successfully")
        except Exception as e:
            print(f"Error initializing audio detector: {str(e)}")
            raise

    def load_model(self, model_path):
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")

            print(f"Loading model from {model_path}")
            model = load_model(model_path, compile=False)
            print("Model loaded successfully")
            return model

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def predict(self, audio_file_path):
        try:
            print(f"Starting audio analysis for {audio_file_path}")
            
            # Load the audio file using soundfile
            try:
                audio, sr = sf.read(audio_file_path)
                if sr != self.target_sr:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
                    sr = self.target_sr
            except Exception as e:
                print(f"Error loading audio file with soundfile: {e}")
                # Fallback to librosa load
                audio, sr = librosa.load(audio_file_path, sr=self.target_sr)
            
            # Extract features (using MFCC exactly as in original code)
            mfccs = librosa.feature.mfcc(y=audio, sr=self.target_sr, n_mfcc=self.n_mfcc)
            
            # Pad or trim to fixed length (exactly as in original code)
            if mfccs.shape[1] < self.max_length:
                mfccs = np.pad(mfccs, ((0, 0), (0, self.max_length - mfccs.shape[1])), mode='constant')
            else:
                mfccs = mfccs[:, :self.max_length]
            
            # Reshape for model input (exactly as in original code)
            features = mfccs.reshape(1, self.n_mfcc, self.max_length, 1)
            
            # Print shape for debugging
            print(f"Input shape: {features.shape}")
            
            # Predict using the model
            prediction = self.model.predict(features, verbose=0)
            
            # Determine result using same threshold as original code
            is_fake = bool(prediction[0][0] > 0.6)
            confidence = float(prediction[0][0] if is_fake else 1 - prediction[0][0]) * 100

            # Get audio duration
            duration = len(audio) / self.target_sr

            result = {
                'result': 'FAKE' if is_fake else 'REAL',
                'confidence': confidence,
                'duration': duration,
                'sample_rate': self.target_sr,
                'input_shape': features.shape[1:].tolist()
            }

            print(f"Audio analysis complete. Result: {result['result']} with {result['confidence']:.2f}% confidence")
            return result

        except Exception as e:
            print(f"Error analyzing audio: {str(e)}")
            raise