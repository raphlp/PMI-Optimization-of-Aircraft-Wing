"""
Bayesian Inference Module for Aerodynamic Coefficient Prediction

Uses Monte Carlo Dropout to estimate prediction uncertainty,
enabling Bayesian optimization of airfoil shapes.

Key features:
- MC Dropout inference for uncertainty quantification
- Prediction intervals (confidence bounds)
- Expected Improvement calculation for Bayesian optimization
"""

import numpy as np
import os


def load_bayesian_model():
    """Load the trained Bayesian CNN model."""
    from tensorflow.keras.models import load_model
    
    model_path = "data/processed/models/cnn_model.keras"
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found. Train the CNN first.")
    
    model = load_model(model_path, compile=False)
    return model


def load_normalization_params():
    """Load saved normalization parameters."""
    models_dir = "data/processed/models"
    
    y_mean_path = os.path.join(models_dir, "y_mean.npy")
    y_std_path = os.path.join(models_dir, "y_std.npy")
    
    if os.path.exists(y_mean_path) and os.path.exists(y_std_path):
        y_mean = np.load(y_mean_path)
        y_std = np.load(y_std_path)
        return y_mean, y_std
    
    return np.array([0, 0]), np.array([1, 1])


def bayesian_predict(model, X, n_samples=50, y_mean=None, y_std=None):
    """
    Perform Bayesian prediction using Monte Carlo Dropout.
    
    Args:
        model: Trained Keras model with MC Dropout (training=True)
        X: Input data (batch_size, h, w, channels)
        n_samples: Number of MC samples (more = better uncertainty estimate)
        y_mean: Mean for denormalization
        y_std: Std for denormalization
    
    Returns:
        mean_pred: Mean prediction (batch_size, 2) [CL, CD]
        std_pred: Standard deviation (batch_size, 2) - uncertainty
        all_preds: All MC samples (n_samples, batch_size, 2)
    """
    if y_mean is None or y_std is None:
        y_mean, y_std = load_normalization_params()
    
    # Collect MC samples
    predictions = []
    for i in range(n_samples):
        # Model with training=True in Dropout layers gives different outputs each time
        pred = model(X, training=True)  # Force dropout active
        predictions.append(pred.numpy())
    
    all_preds = np.array(predictions)  # (n_samples, batch_size, 2)
    
    # Denormalize
    all_preds = all_preds * y_std + y_mean
    
    # Calculate statistics
    mean_pred = np.mean(all_preds, axis=0)  # (batch_size, 2)
    std_pred = np.std(all_preds, axis=0)    # (batch_size, 2)
    
    return mean_pred, std_pred, all_preds


def get_confidence_interval(mean, std, confidence=0.95):
    """
    Calculate confidence interval for predictions.
    
    Args:
        mean: Mean predictions
        std: Standard deviations
        confidence: Confidence level (0.95 = 95%)
    
    Returns:
        lower: Lower bound
        upper: Upper bound
    """
    from scipy import stats
    
    z = stats.norm.ppf((1 + confidence) / 2)
    lower = mean - z * std
    upper = mean + z * std
    
    return lower, upper


def expected_improvement(mean, std, best_value, minimize=True):
    """
    Calculate Expected Improvement for Bayesian optimization.
    
    Args:
        mean: Predicted mean
        std: Predicted standard deviation (uncertainty)
        best_value: Current best observed value
        minimize: If True, we want to minimize (e.g., CD)
                  If False, we want to maximize (e.g., CL/CD ratio)
    
    Returns:
        ei: Expected Improvement score
    """
    from scipy import stats
    
    if minimize:
        improvement = best_value - mean
    else:
        improvement = mean - best_value
    
    # Avoid division by zero
    std = np.maximum(std, 1e-8)
    
    Z = improvement / std
    ei = improvement * stats.norm.cdf(Z) + std * stats.norm.pdf(Z)
    
    return np.maximum(ei, 0)


def predict_single_sample(X_single, n_mc_samples=100):
    """
    Convenience function for single sample prediction with uncertainty.
    
    Args:
        X_single: Single input (h, w, channels) or (1, h, w, channels)
        n_mc_samples: Number of MC samples
    
    Returns:
        dict with prediction results
    """
    model = load_bayesian_model()
    y_mean, y_std = load_normalization_params()
    
    # Ensure batch dimension
    if len(X_single.shape) == 3:
        X_single = X_single[np.newaxis, ...]
    
    mean_pred, std_pred, all_preds = bayesian_predict(
        model, X_single, n_samples=n_mc_samples, 
        y_mean=y_mean, y_std=y_std
    )
    
    lower, upper = get_confidence_interval(mean_pred[0], std_pred[0])
    
    return {
        'CL_mean': float(mean_pred[0, 0]),
        'CL_std': float(std_pred[0, 0]),
        'CL_lower_95': float(lower[0]),
        'CL_upper_95': float(upper[0]),
        'CD_mean': float(mean_pred[0, 1]),
        'CD_std': float(std_pred[0, 1]),
        'CD_lower_95': float(lower[1]),
        'CD_upper_95': float(upper[1]),
    }


def bayesian_optimization_objective(geometry_params, cfd_simulator=None, target='efficiency'):
    """
    Objective function for Bayesian optimization of airfoil shape.
    
    Args:
        geometry_params: Airfoil geometry parameters (e.g., NACA digits, control points)
        cfd_simulator: Function that runs CFD and returns field data
        target: Optimization target ('efficiency' = CL/CD, 'lift', 'drag')
    
    Returns:
        value: Objective value (CL/CD ratio for efficiency)
        uncertainty: Uncertainty estimate
    """
    if cfd_simulator is None:
        raise ValueError("CFD simulator function required")
    
    # 1. Run CFD simulation (or use surrogate)
    field_data = cfd_simulator(geometry_params)
    
    # 2. Predict CL/CD with uncertainty
    result = predict_single_sample(field_data)
    
    # 3. Calculate objective
    if target == 'efficiency':
        # Want to maximize CL/CD (so minimize -CL/CD)
        value = -result['CL_mean'] / (result['CD_mean'] + 1e-8)
        # Propagate uncertainty
        uncertainty = np.sqrt(
            (result['CL_std'] / result['CD_mean'])**2 +
            (result['CL_mean'] * result['CD_std'] / result['CD_mean']**2)**2
        )
    elif target == 'lift':
        value = -result['CL_mean']  # Maximize CL
        uncertainty = result['CL_std']
    elif target == 'drag':
        value = result['CD_mean']  # Minimize CD
        uncertainty = result['CD_std']
    else:
        raise ValueError(f"Unknown target: {target}")
    
    return value, uncertainty


if __name__ == "__main__":
    # Test bayesian prediction
    print("Testing Bayesian Inference Module")
    print("="*50)
    
    try:
        model = load_bayesian_model()
        y_mean, y_std = load_normalization_params()
        
        # Load test data
        X = np.load("data/processed/combined/X.npy")
        
        # Test single prediction
        print(f"\nTesting with sample 0...")
        result = predict_single_sample(X[0])
        
        print(f"\nCL: {result['CL_mean']:.4f} ± {result['CL_std']:.4f}")
        print(f"   95% CI: [{result['CL_lower_95']:.4f}, {result['CL_upper_95']:.4f}]")
        print(f"\nCD: {result['CD_mean']:.6f} ± {result['CD_std']:.6f}")
        print(f"   95% CI: [{result['CD_lower_95']:.6f}, {result['CD_upper_95']:.6f}]")
        
        print("\n✅ Bayesian inference working!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Train the CNN model first.")
