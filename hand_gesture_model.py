# hand_gesture_model.py

"""
Hand Gesture Recognition Model - Single File Version

This script includes:
- Data preprocessing
- Model training
- Gesture prediction

Usage:
1. Train the model:
   python hand_gesture_model.py --mode train --data_dir dataset/

2. Predict a gesture:
   python hand_gesture_model.py --mode predict --image path_to_image.jpg
"""

import os
import cv2
import argparse
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

def load_data(data_dir, img_size=(64, 64)):
    X, y = [], []
    labels = os.listdir(data_dir)
    label_map = {label: idx for idx, label in enumerate(labels)}

    for label in labels:
        path = os.path.join(data_dir, label)
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, img_size)
                X.append(img)
                y.append(label_map[label])

    X = np.array(X) / 255.0
    y = np.array(y)
    return train_test_split(X, y, test_size=0.2), label_map

def build_model(num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train(data_dir):
    (X_train, X_test, y_train, y_test), label_map = load_data(data_dir)
    model = build_model(len(label_map))
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
    model.save("gesture_model.h5")
    print("Model trained and saved as gesture_model.h5")

def predict(image_path):
    model = tf.keras.models.load_model("gesture_model.h5")
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)
    class_id = np.argmax(pred)
    print("Predicted class ID:", class_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "predict"], required=True, help="Mode: train or predict")
    parser.add_argument("--data_dir", help="Path to dataset folder (required for training)")
    parser.add_argument("--image", help="Path to image (required for prediction)")
    args = parser.parse_args()

    if args.mode == "train" and args.data_dir:
        train(args.data_dir)
    elif args.mode == "predict" and args.image:
        predict(args.image)
    else:
        print("Invalid arguments. Use --help for usage.")
