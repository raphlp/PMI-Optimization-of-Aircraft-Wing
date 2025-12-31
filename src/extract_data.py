"""
Extract CFD data from Fluent CSV exports and prepare for CNN training.

Expects the following data structure:
    data/raw/
    ├── NACA23015_AoA0/
    │   ├── fields.csv              # Exported CFD fields (X, Y, P, rho, U, V)
    │   ├── lift_coefficient-rfile.out  # CL convergence history
    │   └── drag-coefficient-rfile.out  # CD convergence history
    ├── NACA23015_AoA5/
    │   └── ...
    └── NACA0012_AoA0/
        └── ...

Each subfolder = 1 simulation case
Naming convention: {PROFILE}_{AoA}{ANGLE}
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy.interpolate import griddata


def read_last_coefficient(path):
    """Reads the last coefficient (CL or CD) from a Fluent .out file"""
    with open(path, "r") as f:
        lines = f.readlines()
    values = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 2:
            try:
                iteration = int(parts[0])
                value = float(parts[1])
                values.append((iteration, value))
            except ValueError:
                continue
    if not values:
        raise ValueError(f"No numeric data found in {path}")
    return values[-1][1]


def load_csv_fields(csv_path):
    """
    Load CFD field data from a Fluent ASCII export CSV.
    
    Expected columns: cellnumber, x-coordinate, y-coordinate, pressure, 
                      density, x-velocity, y-velocity, ...
    """
    df = pd.read_csv(csv_path, skipinitialspace=True)
    df.columns = [col.strip().lower().replace('-', '_') for col in df.columns]
    
    x = df['x_coordinate'].values
    y = df['y_coordinate'].values
    pressure = df['pressure'].values
    density = df['density'].values
    u_velocity = df['x_velocity'].values
    v_velocity = df['y_velocity'].values
    
    return x, y, pressure, density, u_velocity, v_velocity


def interpolate_to_grid(x, y, fields, grid_size=128):
    """
    Interpolate scattered CFD data onto a regular 2D grid.
    """
    xi = np.linspace(x.min(), x.max(), grid_size)
    yi = np.linspace(y.min(), y.max(), grid_size)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    
    grid_fields = []
    points = np.column_stack((x, y))
    
    for field in fields:
        grid_field = griddata(points, field, (xi_grid, yi_grid), method='linear')
        if np.any(np.isnan(grid_field)):
            grid_field_nearest = griddata(points, field, (xi_grid, yi_grid), method='nearest')
            grid_field = np.where(np.isnan(grid_field), grid_field_nearest, grid_field)
        grid_fields.append(grid_field)
    
    return np.stack(grid_fields, axis=-1)


def normalize_fields(data):
    """Normalize each channel to [0, 1] range."""
    normalized = np.zeros_like(data)
    for i in range(data.shape[-1]):
        channel = data[..., i]
        min_val = channel.min()
        max_val = channel.max()
        if max_val - min_val > 1e-10:
            normalized[..., i] = (channel - min_val) / (max_val - min_val)
        else:
            normalized[..., i] = 0.0
    return normalized


def find_cases(data_dir):
    """
    Find all simulation cases in the data directory.
    
    Each case is a subfolder containing:
    - fields.csv (or *_fields* file)
    - *lift*.out and *drag*.out files
    
    Returns list of dicts with case info.
    """
    cases = []
    
    # Look for subfolders
    for folder in sorted(os.listdir(data_dir)):
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        if folder.startswith('.'):
            continue
            
        # Find fields file
        fields_files = glob.glob(os.path.join(folder_path, "fields.csv")) + \
                       glob.glob(os.path.join(folder_path, "*_fields*"))
        
        # Find coefficient files
        lift_files = glob.glob(os.path.join(folder_path, "*lift*.out"))
        drag_files = glob.glob(os.path.join(folder_path, "*drag*.out"))
        
        if fields_files and lift_files and drag_files:
            # Parse case name
            parts = folder.split('_')
            profile = parts[0] if parts else folder
            aoa = 0.0
            for p in parts:
                if p.lower().startswith('aoa'):
                    try:
                        aoa = float(p[3:])
                    except:
                        pass
            
            cases.append({
                'name': folder,
                'profile': profile,
                'aoa': aoa,
                'fields_path': fields_files[0],
                'lift_path': lift_files[0],
                'drag_path': drag_files[0]
            })
    
    return cases


def extract_dataset(data_dir, output_dir, grid_size=128):
    """
    Main extraction function.
    """
    print(f"\n{'='*60}")
    print("CFD Data Extraction Pipeline")
    print(f"{'='*60}")
    print(f"Input directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Grid size: {grid_size}x{grid_size}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all cases
    cases = find_cases(data_dir)
    
    if not cases:
        print("\n⚠️  No valid cases found!")
        print("\nExpected structure:")
        print("  data/raw/")
        print("  └── NACA23015_AoA0/")
        print("      ├── fields.csv")
        print("      ├── lift_coefficient-rfile.out")
        print("      └── drag-coefficient-rfile.out")
        return None, None
    
    print(f"\n✅ Found {len(cases)} case(s):")
    for c in cases:
        print(f"   • {c['name']} (profile={c['profile']}, AoA={c['aoa']}°)")
    
    dataset_X = []
    dataset_y = []
    case_names = []
    
    for i, case in enumerate(cases):
        print(f"\n[{i+1}/{len(cases)}] Processing: {case['name']}")
        
        # Read coefficients
        CL = read_last_coefficient(case['lift_path'])
        CD = read_last_coefficient(case['drag_path'])
        print(f"  → CL = {CL:.6f}, CD = {CD:.6f}")
        
        # Load fields
        x, y, pressure, density, u_vel, v_vel = load_csv_fields(case['fields_path'])
        print(f"  → Loaded {len(x):,} points")
        print(f"  → Velocity magnitude: [{np.sqrt(u_vel**2 + v_vel**2).min():.2f}, {np.sqrt(u_vel**2 + v_vel**2).max():.2f}] m/s")
        
        # Interpolate to grid
        fields = [pressure, density, u_vel, v_vel]
        grid_data = interpolate_to_grid(x, y, fields, grid_size)
        grid_data = normalize_fields(grid_data)
        print(f"  → Grid shape: {grid_data.shape}")
        
        dataset_X.append(grid_data)
        dataset_y.append([CL, CD])
        case_names.append(case['name'])
    
    # Convert to numpy arrays
    X = np.array(dataset_X, dtype=np.float32)
    y = np.array(dataset_y, dtype=np.float32)
    
    # Save
    np.save(os.path.join(output_dir, "CFD_X.npy"), X)
    np.save(os.path.join(output_dir, "CFD_y.npy"), y)
    
    # Save case names for reference
    with open(os.path.join(output_dir, "case_names.txt"), 'w') as f:
        for name in case_names:
            f.write(name + '\n')
    
    print(f"\n{'='*60}")
    print("Extraction Complete!")
    print(f"{'='*60}")
    print(f"  → X shape: {X.shape} (samples, height, width, channels)")
    print(f"  → y shape: {y.shape} (samples, [CL, CD])")
    print(f"  → Channels: [pressure, density, u_velocity, v_velocity]")
    print(f"  → Cases: {case_names}")
    print(f"  → Files saved to: {output_dir}")
    
    return X, y


if __name__ == "__main__":
    data_dir = os.path.join("data", "raw")
    output_dir = os.path.join("data", "processed")
    extract_dataset(data_dir, output_dir, grid_size=128)
