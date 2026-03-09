"""
Training script for the audio deepfake detection model (my_model.h5).

Requirements:
    pip install tensorflow keras librosa soundfile numpy scikit-learn

Dataset:
    Download the "Fake or Real" dataset from Kaggle:
    https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset

    Extract it so the folder structure looks like:
        dataset/
        ├── real/       (real audio .wav files)
        └── fake/       (fake audio .wav files)

Usage:
    python train_audio_model.py --dataset_path ./dataset

    This will produce my_model.h5 in the same directory.
"""

import os
import argparse
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

N_MFCC = 40
MAX_LENGTH = 500
TARGET_SR = 16000


def extract_mfcc(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=TARGET_SR)
        mfccs = librosa.feature.mfcc(y=audio, sr=TARGET_SR, n_mfcc=N_MFCC)

        if mfccs.shape[1] < MAX_LENGTH:
            mfccs = np.pad(mfccs, ((0, 0), (0, MAX_LENGTH - mfccs.shape[1])), mode='constant')
        else:
            mfccs = mfccs[:, :MAX_LENGTH]

        return mfccs
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def load_dataset(dataset_path):
    features = []
    labels = []

    real_dir = os.path.join(dataset_path, 'real')
    fake_dir = os.path.join(dataset_path, 'fake')

    if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
        for name in os.listdir(dataset_path):
            lower = name.lower()
            if 'real' in lower or 'genuine' in lower or 'bonafide' in lower:
                real_dir = os.path.join(dataset_path, name)
            elif 'fake' in lower or 'spoof' in lower or 'synthetic' in lower:
                fake_dir = os.path.join(dataset_path, name)

    print(f"Real audio dir: {real_dir}")
    print(f"Fake audio dir: {fake_dir}")

    for label, folder in [(0, real_dir), (1, fake_dir)]:
        label_name = "REAL" if label == 0 else "FAKE"
        files = [f for f in os.listdir(folder) if f.endswith(('.wav', '.flac', '.mp3', '.ogg'))]
        print(f"Loading {len(files)} {label_name} files...")

        for i, fname in enumerate(files):
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(files)}")
            mfcc = extract_mfcc(os.path.join(folder, fname))
            if mfcc is not None:
                features.append(mfcc)
                labels.append(label)

    X = np.array(features).reshape(-1, N_MFCC, MAX_LENGTH, 1)
    y = np.array(labels)
    print(f"Dataset loaded: {X.shape[0]} samples ({sum(y == 0)} real, {sum(y == 1)} fake)")
    return X, y


def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(N_MFCC, MAX_LENGTH, 1)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid'),
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def main():
    parser = argparse.ArgumentParser(description='Train audio deepfake detection model')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to dataset with real/ and fake/ folders')
    parser.add_argument('--epochs', type=int, default=30, help='Max training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    args = parser.parse_args()

    print("Loading dataset...")
    X, y = load_dataset(args.dataset_path)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

    print("Building model...")
    model = build_model()
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
    ]

    print("Training...")
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
    )

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {accuracy:.4f}")

    output_path = os.path.join(os.path.dirname(__file__), 'my_model.h5')
    model.save(output_path)
    print(f"Model saved to {output_path}")


if __name__ == '__main__':
    main()
