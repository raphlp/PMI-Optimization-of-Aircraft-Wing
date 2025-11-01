import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def load_dataset(processed_dir):
    X = np.load(os.path.join(processed_dir, "CFD_X.npy"))
    y = np.load(os.path.join(processed_dir, "CFD_y.npy"))

    # Reduce temporarily to avoid memory issues
    X = X[:, :1000, :]  # keep only 1000 points per case for testing
    X = (X - np.min(X)) / (np.max(X) - np.min(X))
    return X, y

def build_cnn_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Reshape((input_shape[0], input_shape[1], 1)),  # dynamic reshape
        layers.Conv2D(8, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(2, activation="linear")
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

def train_cnn_model(processed_dir):
    print("\n--- Training CNN Model ---")
    X, y = load_dataset(processed_dir)
    print(f"Dataset loaded: X={X.shape}, y={y.shape}")

    input_shape = X.shape[1:]
    model = build_cnn_model(input_shape)

    history = model.fit(X, y, epochs=5, batch_size=1, validation_split=0.2)
    model.save(os.path.join(processed_dir, "cnn_model.h5"))
    print("Model saved to data/processed/cnn_model.h5")
