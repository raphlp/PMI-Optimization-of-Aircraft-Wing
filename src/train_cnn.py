"""
CNN model for predicting aerodynamic coefficients (CL, CD) from CFD field data.

Architecture:
- Input: 2D grid images with 4 channels (pressure, density, u_velocity, v_velocity)
- Convolutional layers to extract spatial features
- Dense layers for regression to CL/CD
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


def load_dataset(processed_dir):
    """Load preprocessed CFD dataset."""
    X = np.load(os.path.join(processed_dir, "CFD_X.npy"))
    y = np.load(os.path.join(processed_dir, "CFD_y.npy"))
    
    print(f"Dataset loaded:")
    print(f"  → X shape: {X.shape}")
    print(f"  → y shape: {y.shape}")
    print(f"  → X range: [{X.min():.4f}, {X.max():.4f}]")
    print(f"  → y (CL, CD): {y}")
    
    return X, y


def build_cnn_model(input_shape):
    """
    Build CNN architecture for CFD field regression.
    
    Args:
        input_shape: (height, width, channels) - typically (128, 128, 4)
    
    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # First conv block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second conv block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third conv block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fourth conv block
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        
        # Dense layers for regression
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dense(2, activation='linear')  # Output: [CL, CD]
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model


def train_cnn_model(processed_dir, epochs=50, batch_size=8):
    """
    Train the CNN model on CFD data.
    
    Args:
        processed_dir: Directory containing CFD_X.npy and CFD_y.npy
        epochs: Number of training epochs
        batch_size: Training batch size
    """
    print("\n" + "="*60)
    print("CNN Training Pipeline")
    print("="*60)
    
    # Load data
    X, y = load_dataset(processed_dir)
    
    # Handle small datasets (data augmentation for single sample)
    if len(X) == 1:
        print("\nWarning: Only 1 sample available. Creating augmented copies for training demo...")
        # Create augmented versions for demonstration
        X_aug = [X[0]]
        y_aug = [y[0]]
        
        # Add noise variations
        for i in range(19):
            noise = np.random.normal(0, 0.02, X[0].shape)
            X_aug.append(np.clip(X[0] + noise, 0, 1))
            # Slightly vary the targets too
            y_noise = np.random.normal(0, 0.001, y[0].shape)
            y_aug.append(y[0] + y_noise)
        
        X = np.array(X_aug, dtype=np.float32)
        y = np.array(y_aug, dtype=np.float32)
        print(f"Augmented dataset: X={X.shape}, y={y.shape}")
    
    # Build model
    input_shape = X.shape[1:]  # (height, width, channels)
    model = build_cnn_model(input_shape)
    
    print("\nModel Summary:")
    model.summary()
    
    # Callbacks
    model_callbacks = [
        callbacks.EarlyStopping(
            monitor='loss',
            patience=10,
            restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]
    
    # Train
    print("\nTraining...")
    history = model.fit(
        X, y,
        epochs=epochs,
        batch_size=min(batch_size, len(X)),
        validation_split=0.2 if len(X) > 5 else 0.0,
        callbacks=model_callbacks,
        verbose=1
    )
    
    # Save model
    model_path = os.path.join(processed_dir, "cnn_model.keras")
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Evaluate
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    
    # Make predictions on training data
    predictions = model.predict(X[:5], verbose=0)
    print("\nSample Predictions vs Actual:")
    print(f"{'Sample':<10} {'Pred CL':<12} {'Actual CL':<12} {'Pred CD':<12} {'Actual CD':<12}")
    print("-" * 58)
    for i in range(min(5, len(X))):
        print(f"{i+1:<10} {predictions[i, 0]:<12.6f} {y[i, 0]:<12.6f} {predictions[i, 1]:<12.6f} {y[i, 1]:<12.6f}")
    
    return model, history


if __name__ == "__main__":
    processed_dir = os.path.join("data", "processed")
    train_cnn_model(processed_dir, epochs=50)
