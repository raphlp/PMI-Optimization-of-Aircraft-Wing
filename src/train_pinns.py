"""
PINNs Training Pipeline

High-level interface for training Physics-Informed Neural Networks
for aerodynamic flow prediction.
"""

import os
import numpy as np
import tensorflow as tf

from src.models.naca_geometry import generate_domain_points, generate_naca_profile
from src.models.pinns_model import create_pinn_model


def load_cfd_data(data_dir):
    """
    Load CFD data for hybrid training mode.
    Normalizes data to comparable scales.
    
    Returns:
        cfd_data dict or None if no data available
    """
    import pandas as pd
    import glob
    
    # Find CFD files
    cases = []
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path) and not folder.startswith('.'):
            fields_files = glob.glob(os.path.join(folder_path, "fields.csv")) + \
                           glob.glob(os.path.join(folder_path, "*_fields*"))
            if fields_files:
                cases.append(fields_files[0])
    
    if not cases:
        return None
    
    # Load first case
    df = pd.read_csv(cases[0], skipinitialspace=True)
    df.columns = [col.strip().lower().replace('-', '_') for col in df.columns]
    
    # Sample a subset for training (too many points is slow)
    if len(df) > 10000:
        df = df.sample(n=10000, random_state=42)
    
    # Extract raw values
    x = df['x_coordinate'].values
    y = df['y_coordinate'].values
    u = df['x_velocity'].values
    v = df['y_velocity'].values
    p = df['pressure'].values
    
    # Normalize velocities by freestream velocity (~40 m/s typical)
    U_ref = max(np.max(np.abs(u)), 1.0)
    u_norm = u / U_ref
    v_norm = v / U_ref
    
    # Normalize pressure to zero mean, unit variance
    p_mean = np.mean(p)
    p_std = np.std(p) if np.std(p) > 0 else 1.0
    p_norm = (p - p_mean) / p_std
    
    cfd_data = {
        'x': tf.constant(x, dtype=tf.float32),
        'y': tf.constant(y, dtype=tf.float32),
        'u': tf.constant(u_norm, dtype=tf.float32),
        'v': tf.constant(v_norm, dtype=tf.float32),
        'p': tf.constant(p_norm, dtype=tf.float32),
        # Store normalization params for denormalization
        'U_ref': U_ref,
        'p_mean': p_mean,
        'p_std': p_std
    }
    
    return cfd_data


def train_pinns(naca_code="23015", angle_of_attack=0.0, epochs=2000,
                use_cfd_data=False, data_dir="data/raw", output_dir="data/processed",
                hidden_layers=[64, 64, 64, 64], learning_rate=1e-3,
                n_domain_points=5000, callback=None):
    """
    Train a PINN model for airfoil flow prediction.
    
    Args:
        naca_code: NACA airfoil designation
        angle_of_attack: Angle of attack in degrees
        epochs: Number of training epochs
        use_cfd_data: Whether to use CFD data (hybrid mode)
        data_dir: Directory containing CFD data
        output_dir: Directory to save trained model
        hidden_layers: Neural network architecture
        learning_rate: Optimizer learning rate
        n_domain_points: Number of collocation points
        callback: Optional progress callback
    
    Returns:
        model, trainer, history
    """
    print("\n" + "="*60)
    print("PINN Training Pipeline")
    print("="*60)
    print(f"Profile: NACA {naca_code}")
    print(f"Angle of Attack: {angle_of_attack} deg")
    print(f"Mode: {'Hybrid (with CFD)' if use_cfd_data else 'Pure Physics'}")
    print(f"Epochs: {epochs}")
    print(f"Domain Points: {n_domain_points}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate domain and boundary points
    print("\nGenerating collocation points...")
    domain_points, boundary_points, inlet_points, outlet_points = generate_domain_points(
        naca_code=naca_code,
        n_domain=n_domain_points,
        n_boundary=200,
        x_min=-0.5,
        x_max=2.0,
        y_min=-1.0,
        y_max=1.0,
        chord=1.0
    )
    
    print(f"  Domain points: {len(domain_points)}")
    print(f"  Boundary points: {len(boundary_points)}")
    print(f"  Inlet points: {len(inlet_points)}")
    
    # Load CFD data if requested
    cfd_data = None
    if use_cfd_data:
        print("\nLoading CFD data...")
        cfd_data = load_cfd_data(data_dir)
        if cfd_data:
            print(f"  Loaded {len(cfd_data['x'])} CFD data points")
        else:
            print("  Warning: No CFD data found, using pure physics mode")
    
    # Create model
    print("\nCreating PINN model...")
    model, trainer = create_pinn_model(
        hidden_layers=hidden_layers,
        learning_rate=learning_rate
    )
    
    # Build model
    test_input = tf.constant([[0.0, 0.0]], dtype=tf.float32)
    _ = model(test_input)
    print(f"  Parameters: {model.count_params():,}")
    
    # Train
    print("\nTraining...")
    history = trainer.train(
        domain_points=domain_points,
        boundary_points=boundary_points,
        inlet_points=inlet_points,
        angle_of_attack=angle_of_attack,
        epochs=epochs,
        cfd_data=cfd_data,
        batch_size=1000,
        callback=callback
    )
    
    # Save model
    model_path = os.path.join(output_dir, "pinns_model.keras")
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Generate predictions on a grid for visualization
    print("\nGenerating flow field predictions...")
    # Use 128x128 grid to match CFD processed data format
    x_grid = np.linspace(-0.5, 2.0, 128)
    y_grid = np.linspace(-1.0, 1.0, 128)
    X, Y = np.meshgrid(x_grid, y_grid)
    x_flat = X.flatten().astype(np.float32)
    y_flat = Y.flatten().astype(np.float32)
    
    u_pred, v_pred, p_pred = trainer.predict(x_flat, y_flat)
    
    # Save predictions
    np.savez(
        os.path.join(output_dir, "pinns_predictions.npz"),
        x=X, y=Y,
        u=u_pred.reshape(X.shape),
        v=v_pred.reshape(X.shape),
        p=p_pred.reshape(X.shape)
    )
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"  Final Loss: {history['loss'][-1]:.6f}")
    print(f"  Physics Loss: {history['physics_loss'][-1]:.6f}")
    print(f"  BC Loss: {history['bc_loss'][-1]:.6f}")
    
    return model, trainer, history


if __name__ == "__main__":
    # Test training
    model, trainer, history = train_pinns(
        naca_code="23015",
        angle_of_attack=0.0,
        epochs=500,
        use_cfd_data=False
    )
