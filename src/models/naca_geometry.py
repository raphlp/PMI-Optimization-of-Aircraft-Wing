"""
NACA Airfoil Geometry Generator

Generates NACA 4-digit and 5-digit airfoil profiles mathematically.
No external files needed.
"""

import numpy as np


def naca4_thickness(x, t):
    """
    Calculate NACA 4-digit thickness distribution.
    
    Args:
        x: Normalized chord position (0 to 1)
        t: Maximum thickness as fraction of chord
    
    Returns:
        Half-thickness at position x
    """
    return 5 * t * (
        0.2969 * np.sqrt(x) 
        - 0.1260 * x 
        - 0.3516 * x**2 
        + 0.2843 * x**3 
        - 0.1015 * x**4
    )


def naca4_camber(x, m, p):
    """
    Calculate NACA 4-digit camber line.
    
    Args:
        x: Normalized chord position (0 to 1)
        m: Maximum camber as fraction of chord
        p: Position of maximum camber as fraction of chord
    
    Returns:
        Camber line y-coordinate
    """
    yc = np.zeros_like(x)
    
    if p > 0:
        front = x < p
        back = x >= p
        
        yc[front] = (m / p**2) * (2*p*x[front] - x[front]**2)
        yc[back] = (m / (1-p)**2) * ((1 - 2*p) + 2*p*x[back] - x[back]**2)
    
    return yc


def generate_naca_profile(naca_code="23015", n_points=200, chord=1.0):
    """
    Generate NACA airfoil coordinates.
    
    Args:
        naca_code: NACA designation (e.g., "0012", "2412", "23015")
        n_points: Number of points on each surface
        chord: Chord length
    
    Returns:
        x_upper, y_upper, x_lower, y_lower: Coordinate arrays
    """
    # Parse NACA code
    if len(naca_code) == 4:
        m = int(naca_code[0]) / 100  # Max camber
        p = int(naca_code[1]) / 10   # Camber position
        t = int(naca_code[2:4]) / 100  # Thickness
    elif len(naca_code) == 5:
        # NACA 5-digit (simplified)
        m = int(naca_code[0]) * 0.15 / 6  # Approximate
        p = int(naca_code[1]) / 20
        t = int(naca_code[3:5]) / 100
    else:
        raise ValueError(f"Unsupported NACA code: {naca_code}")
    
    # Generate x coordinates (cosine spacing for better leading edge resolution)
    beta = np.linspace(0, np.pi, n_points)
    x = 0.5 * (1 - np.cos(beta))
    
    # Calculate thickness and camber
    yt = naca4_thickness(x, t)
    yc = naca4_camber(x, m, p)
    
    # Calculate camber line slope for proper thickness distribution
    dyc = np.zeros_like(x)
    if p > 0:
        front = x < p
        back = x >= p
        dyc[front] = (2*m / p**2) * (p - x[front])
        dyc[back] = (2*m / (1-p)**2) * (p - x[back])
    
    theta = np.arctan(dyc)
    
    # Upper and lower surface coordinates
    x_upper = (x - yt * np.sin(theta)) * chord
    y_upper = (yc + yt * np.cos(theta)) * chord
    x_lower = (x + yt * np.sin(theta)) * chord
    y_lower = (yc - yt * np.cos(theta)) * chord
    
    return x_upper, y_upper, x_lower, y_lower


def get_airfoil_boundary_points(naca_code="23015", n_points=200, chord=1.0):
    """
    Get all boundary points of the airfoil as a single array.
    
    Returns:
        points: Array of shape (N, 2) with (x, y) coordinates
    """
    x_upper, y_upper, x_lower, y_lower = generate_naca_profile(naca_code, n_points, chord)
    
    # Combine upper and lower (reverse lower to make continuous)
    x_all = np.concatenate([x_upper, x_lower[::-1][1:]])
    y_all = np.concatenate([y_upper, y_lower[::-1][1:]])
    
    return np.column_stack([x_all, y_all])


def generate_domain_points(naca_code="23015", n_domain=5000, n_boundary=200,
                           x_min=-0.5, x_max=2.0, y_min=-1.0, y_max=1.0, chord=1.0):
    """
    Generate collocation points for PINN training.
    
    Returns:
        domain_points: Points inside the fluid domain (N, 2)
        boundary_points: Points on airfoil surface (M, 2)
        inlet_points: Points at inlet (K, 2)
        outlet_points: Points at outlet (L, 2)
    """
    # Airfoil boundary
    boundary_points = get_airfoil_boundary_points(naca_code, n_boundary, chord)
    
    # Random domain points
    x_domain = np.random.uniform(x_min, x_max, n_domain)
    y_domain = np.random.uniform(y_min, y_max, n_domain)
    domain_points = np.column_stack([x_domain, y_domain])
    
    # Remove points inside the airfoil (simplified check)
    x_upper, y_upper, x_lower, y_lower = generate_naca_profile(naca_code, 100, chord)
    
    # Create interpolation functions for upper and lower surfaces
    from scipy.interpolate import interp1d
    upper_interp = interp1d(x_upper, y_upper, bounds_error=False, fill_value=0)
    lower_interp = interp1d(x_lower, y_lower, bounds_error=False, fill_value=0)
    
    # Filter out points inside airfoil
    mask = np.ones(len(domain_points), dtype=bool)
    for i, (x, y) in enumerate(domain_points):
        if 0 <= x <= chord:
            y_up = upper_interp(x)
            y_lo = lower_interp(x)
            if y_lo <= y <= y_up:
                mask[i] = False
    
    domain_points = domain_points[mask]
    
    # Inlet points (left boundary)
    y_inlet = np.linspace(y_min, y_max, 100)
    inlet_points = np.column_stack([np.full_like(y_inlet, x_min), y_inlet])
    
    # Outlet points (right boundary)
    y_outlet = np.linspace(y_min, y_max, 100)
    outlet_points = np.column_stack([np.full_like(y_outlet, x_max), y_outlet])
    
    return domain_points, boundary_points, inlet_points, outlet_points


if __name__ == "__main__":
    # Test
    x_up, y_up, x_lo, y_lo = generate_naca_profile("23015")
    print(f"Generated NACA 23015: {len(x_up)} points per surface")
    print(f"Leading edge: ({x_up[0]:.4f}, {y_up[0]:.4f})")
    print(f"Trailing edge: ({x_up[-1]:.4f}, {y_up[-1]:.4f})")
