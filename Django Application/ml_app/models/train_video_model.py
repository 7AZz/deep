"""
Training script for the video/image deepfake detection model (cnn_model.h5).

Requirements:
    pip install tensorflow keras opencv-python numpy scikit-learn mtcnn

Datasets (pick one):
    1. 140k Real and Fake Faces (Kaggle) - RECOMMENDED, ~1.5 GB
       https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces

    2. Celeb-DF v2 (GitHub) - Celebrity deepfake videos
       https://github.com/yuezunli/celeb-deepfakeforensics

    3. FaceForensics++ (request access)
       https://github.com/ondyari/FaceForensics

    For image datasets, organize as:
        dataset/
        ├── real/       (real face images)
        └── fake/       (deepfake face images)

    For video datasets, organize as:
        dataset/
        ├── real/       (real videos)
        └── fake/       (deepfake videos)

Usage:
    Image dataset:
        python train_video_model.py --dataset_path ./dataset --mode images

    Video dataset (extracts frames + detects faces automatically):
        python train_video_model.py --dataset_path ./dataset --mode videos

    This will produce cnn_model.h5 in the same directory.
"""

import os
import argparse
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

TARGET_SIZE = (128, 128)
FACE_DETECTOR = None


def get_face_detector():
    global FACE_DETECTOR
    if FACE_DETECTOR is None:
        from mtcnn import MTCNN
        FACE_DETECTOR = MTCNN()
    return FACE_DETECTOR


def detect_and_crop_face(frame):
    detector = get_face_detector()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(frame_rgb)

    if results:
        x, y, w, h = results[0]['box']
        padding = int(min(w, h) * 0.1)
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(frame.shape[1] - x, w + 2 * padding)
        h = min(frame.shape[0] - y, h + 2 * padding)
        face = frame[y:y+h, x:x+w]
        return cv2.resize(face, TARGET_SIZE), True

    return cv2.resize(frame, TARGET_SIZE), False


def extract_frames_from_video(video_path, max_frames=16):
    frames = []
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return frames

    indices = np.linspace(0, total - 1, min(max_frames, total), dtype=int)
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            face, detected = detect_and_crop_face(frame)
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            frames.append(face_rgb)
    cap.release()
    return frames


def load_images_from_dir(folder, label):
    features, labels = [], []
    files = [f for f in os.listdir(folder)
             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
    print(f"Loading {len(files)} images (label={'FAKE' if label else 'REAL'})...")

    for i, fname in enumerate(files):
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(files)}")
        img = cv2.imread(os.path.join(folder, fname))
        if img is None:
            continue
        img = cv2.resize(img, TARGET_SIZE)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        features.append(img_rgb)
        labels.append(label)

    return features, labels


def load_videos_from_dir(folder, label, max_frames_per_video=16):
    features, labels = [], []
    files = [f for f in os.listdir(folder)
             if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))]
    print(f"Loading {len(files)} videos (label={'FAKE' if label else 'REAL'})...")

    for i, fname in enumerate(files):
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(files)}")
        frames = extract_frames_from_video(
            os.path.join(folder, fname), max_frames_per_video
        )
        for frame in frames:
            features.append(frame)
            labels.append(label)

    return features, labels


def resolve_dirs(dataset_path):
    real_dir = os.path.join(dataset_path, 'real')
    fake_dir = os.path.join(dataset_path, 'fake')

    if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
        for name in os.listdir(dataset_path):
            lower = name.lower()
            if 'real' in lower or 'genuine' in lower or 'authentic' in lower:
                real_dir = os.path.join(dataset_path, name)
            elif 'fake' in lower or 'deepfake' in lower or 'synthetic' in lower:
                fake_dir = os.path.join(dataset_path, name)

    return real_dir, fake_dir


def load_dataset(dataset_path, mode):
    real_dir, fake_dir = resolve_dirs(dataset_path)
    print(f"Real dir: {real_dir}")
    print(f"Fake dir: {fake_dir}")

    if mode == 'images':
        real_features, real_labels = load_images_from_dir(real_dir, label=0)
        fake_features, fake_labels = load_images_from_dir(fake_dir, label=1)
    else:
        real_features, real_labels = load_videos_from_dir(real_dir, label=0)
        fake_features, fake_labels = load_videos_from_dir(fake_dir, label=1)

    all_features = real_features + fake_features
    all_labels = real_labels + fake_labels

    X = np.array(all_features, dtype=np.float32) / 255.0
    y = np.array(all_labels, dtype=np.float32)

    X, y = shuffle(X, y, random_state=42)
    print(f"Dataset: {len(X)} samples ({sum(y == 0):.0f} real, {sum(y == 1):.0f} fake)")
    return X, y


def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(128, 128, 3)),
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

        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid'),
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def main():
    parser = argparse.ArgumentParser(description='Train video/image deepfake detection model')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to dataset with real/ and fake/ folders')
    parser.add_argument('--mode', type=str, choices=['images', 'videos'], default='images',
                        help='Dataset type: "images" for face images, "videos" for video files')
    parser.add_argument('--epochs', type=int, default=50, help='Max training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--max_frames', type=int, default=16,
                        help='Max frames to extract per video (only for video mode)')
    args = parser.parse_args()

    print("Loading dataset...")
    X, y = load_dataset(args.dataset_path, args.mode)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")

    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
    )
    datagen.fit(X_train)

    print("Building model...")
    model = build_model()
    model.summary()

    output_path = os.path.join(os.path.dirname(__file__), 'cnn_model.h5')

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6),
        ModelCheckpoint(output_path, monitor='val_accuracy', save_best_only=True, verbose=1),
    ]

    print("Training with data augmentation...")
    model.fit(
        datagen.flow(X_train, y_train, batch_size=args.batch_size),
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        callbacks=callbacks,
    )

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\nFinal test accuracy: {accuracy:.4f}")
    print(f"Final test loss: {loss:.4f}")

    model.save(output_path)
    print(f"Model saved to {output_path}")


if __name__ == '__main__':
    main()
