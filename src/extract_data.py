"""
CFD Data Extraction Module

Extracts field data from ANSYS Fluent simulation exports
and organizes into structured format for CNN training.

New structure:
  data/processed/
  ├── cfd/
  │   └── {PROFILE}/
  │       └── AoA_{ANGLE}/
  │           ├── fields.npy
  │           └── info.json
  ├── combined/
  │   ├── X.npy
  │   ├── y.npy
  │   └── manifest.json
  └── models/
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.interpolate import griddata


def parse_case_folder(folder_name):
    """Parse folder name to extract profile and angle info."""
    name = folder_name.upper()
    
    # Extract NACA code
    naca_code = "Unknown"
    if "NACA" in name:
        parts = name.split("NACA")
        if len(parts) > 1:
            code_part = parts[1].split("_")[0].split("-")[0]
            naca_code = ''.join(filter(str.isdigit, code_part[:5]))
            if naca_code:
                naca_code = f"NACA{naca_code}"
    
    # Extract angle
    aoa = 0.0
    if "AOA" in name:
        for part in name.replace("_", " ").replace("-", " ").split():
            if "AOA" in part.upper():
                try:
                    aoa = float(''.join(filter(lambda x: x.isdigit() or x == '.' or x == '-', part.replace("AOA", ""))))
                except:
                    pass
    
    return naca_code, aoa


def load_fluent_data(folder_path):
    """Load field data from Fluent export."""
    fields_file = None
    lift_file = None
    drag_file = None
    
    for f in os.listdir(folder_path):
        f_lower = f.lower()
        full_path = os.path.join(folder_path, f)
        
        # Skip directories
        if os.path.isdir(full_path):
            continue
        
        # Field data files (CSV or files with 'field' in name)
        if f_lower.endswith('.csv'):
            fields_file = full_path
        elif 'field' in f_lower and not f_lower.endswith('.out'):
            # Try to use as CSV even without extension
            fields_file = full_path
        
        # Fluent coefficient output files
        if 'lift' in f_lower and f_lower.endswith('.out'):
            lift_file = full_path
        if 'drag' in f_lower and f_lower.endswith('.out'):
            drag_file = full_path
    
    if not fields_file:
        raise FileNotFoundError(f"No field data file found in {folder_path}")
    
    # Load field data
    df = pd.read_csv(fields_file, skipinitialspace=True)
    df.columns = [col.strip().lower().replace('-', '_') for col in df.columns]
    
    # Load coefficients from Fluent .out files (convergence history)
    cl, cd = None, None
    
    if lift_file and os.path.exists(lift_file):
        cl = read_fluent_coefficient(lift_file)
    
    if drag_file and os.path.exists(drag_file):
        cd = read_fluent_coefficient(drag_file)
    
    return df, cl, cd


def read_fluent_coefficient(filepath):
    """Read last converged value from Fluent coefficient output file."""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Find lines with numeric data (iteration value)
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    # Format: iteration coefficient
                    iteration = int(parts[0])
                    value = float(parts[1])
                    return value
                except ValueError:
                    continue
        return None
    except Exception:
        return None


def interpolate_to_grid(df, grid_size=128):
    """Interpolate scattered data to regular grid."""
    x = df['x_coordinate'].values
    y = df['y_coordinate'].values
    
    # Define grid bounds
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    # Create regular grid
    xi = np.linspace(x_min, x_max, grid_size)
    yi = np.linspace(y_min, y_max, grid_size)
    Xi, Yi = np.meshgrid(xi, yi)
    
    points = np.column_stack((x, y))
    
    # Interpolate each field
    p = griddata(points, df['pressure'].values, (Xi, Yi), method='linear', fill_value=0)
    u = griddata(points, df['x_velocity'].values, (Xi, Yi), method='linear', fill_value=0)
    v = griddata(points, df['y_velocity'].values, (Xi, Yi), method='linear', fill_value=0)
    vel_mag = np.sqrt(u**2 + v**2)
    
    # Normalize each channel
    fields = []
    for field in [p, u, v, vel_mag]:
        f_min, f_max = field.min(), field.max()
        if f_max - f_min > 1e-10:
            field = (field - f_min) / (f_max - f_min)
        fields.append(field)
    
    return np.stack(fields, axis=-1)


def extract_single_case(folder_path, output_dir, grid_size=128):
    """Extract a single case to organized structure."""
    folder_name = os.path.basename(folder_path)
    profile, aoa = parse_case_folder(folder_name)
    
    print(f"  Processing: {folder_name}")
    print(f"    Profile: {profile}, AoA: {aoa}°")
    
    # Load data
    df, cl, cd = load_fluent_data(folder_path)
    
    # Interpolate to grid
    fields = interpolate_to_grid(df, grid_size)
    
    # Create output directory - use original folder name for Unknown profiles
    # to prevent overwriting when multiple cases have same profile/angle
    if profile == "Unknown":
        case_id = folder_name  # Use original name like "airfrans_0001"
        case_dir = os.path.join(output_dir, "cfd", case_id, f"AoA_{aoa}")
    else:
        case_dir = os.path.join(output_dir, "cfd", profile, f"AoA_{aoa}")
    os.makedirs(case_dir, exist_ok=True)
    
    # Save fields
    np.save(os.path.join(case_dir, "fields.npy"), fields)
    
    # Save metadata
    info = {
        "source": "CFD",
        "profile": profile,
        "angle_of_attack": aoa,
        "original_folder": folder_name,
        "grid_size": grid_size,
        "cl": cl,
        "cd": cd,
        "num_points": len(df)
    }
    with open(os.path.join(case_dir, "info.json"), 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"    CL: {cl}, CD: {cd}")
    
    return profile, aoa, fields, cl, cd


def build_combined_dataset(processed_dir):
    """Build combined dataset from all processed data."""
    cfd_dir = os.path.join(processed_dir, "cfd")
    pinns_dir = os.path.join(processed_dir, "pinns")
    combined_dir = os.path.join(processed_dir, "combined")
    os.makedirs(combined_dir, exist_ok=True)
    
    X_list = []
    y_list = []
    manifest = []
    
    # Process CFD data
    if os.path.exists(cfd_dir):
        for profile in os.listdir(cfd_dir):
            profile_dir = os.path.join(cfd_dir, profile)
            if not os.path.isdir(profile_dir):
                continue
            
            for angle_folder in os.listdir(profile_dir):
                angle_dir = os.path.join(profile_dir, angle_folder)
                if not os.path.isdir(angle_dir):
                    continue
                
                fields_path = os.path.join(angle_dir, "fields.npy")
                info_path = os.path.join(angle_dir, "info.json")
                
                if os.path.exists(fields_path) and os.path.exists(info_path):
                    fields = np.load(fields_path)
                    with open(info_path, 'r') as f:
                        info = json.load(f)
                    
                    X_list.append(fields)
                    
                    cl = info.get('cl')
                    cd = info.get('cd')
                    if cl is not None and cd is not None:
                        y_list.append([cl, cd])
                    else:
                        y_list.append([np.nan, np.nan])
                    
                    manifest.append({
                        "index": len(manifest),
                        "source": "CFD",
                        "profile": profile,
                        "angle": info.get('angle_of_attack', 0),
                        "path": angle_dir
                    })
    
    # Process PINN data
    if os.path.exists(pinns_dir):
        for profile in os.listdir(pinns_dir):
            profile_dir = os.path.join(pinns_dir, profile)
            if not os.path.isdir(profile_dir):
                continue
            
            for angle_folder in os.listdir(profile_dir):
                angle_dir = os.path.join(profile_dir, angle_folder)
                if not os.path.isdir(angle_dir):
                    continue
                
                fields_path = os.path.join(angle_dir, "fields.npy")
                info_path = os.path.join(angle_dir, "info.json")
                
                if os.path.exists(fields_path):
                    fields = np.load(fields_path)
                    
                    X_list.append(fields)
                    y_list.append([np.nan, np.nan])  # PINN has no ground truth
                    
                    info = {}
                    if os.path.exists(info_path):
                        with open(info_path, 'r') as f:
                            info = json.load(f)
                    
                    manifest.append({
                        "index": len(manifest),
                        "source": "PINN",
                        "profile": profile,
                        "angle": info.get('angle_of_attack', 0),
                        "path": angle_dir
                    })
    
    if X_list:
        X = np.stack(X_list, axis=0)
        y = np.array(y_list)
        
        np.save(os.path.join(combined_dir, "X.npy"), X)
        np.save(os.path.join(combined_dir, "y.npy"), y)
        
        with open(os.path.join(combined_dir, "manifest.json"), 'w') as f:
            json.dump({
                "total_samples": len(manifest),
                "cfd_samples": sum(1 for m in manifest if m['source'] == 'CFD'),
                "pinn_samples": sum(1 for m in manifest if m['source'] == 'PINN'),
                "samples": manifest
            }, f, indent=2)
        
        print(f"\nCombined dataset built:")
        print(f"  Total: {len(manifest)} samples")
        print(f"  CFD: {sum(1 for m in manifest if m['source'] == 'CFD')}")
        print(f"  PINN: {sum(1 for m in manifest if m['source'] == 'PINN')}")
        
        return X, y
    
    return None, None


def extract_dataset(raw_dir, processed_dir, grid_size=128):
    """Extract all CFD cases and build combined dataset."""
    print("=" * 60)
    print("CFD Data Extraction")
    print("=" * 60)
    
    if not os.path.exists(raw_dir):
        print(f"Error: {raw_dir} not found")
        return None, None
    
    cases = []
    for folder in os.listdir(raw_dir):
        folder_path = os.path.join(raw_dir, folder)
        if os.path.isdir(folder_path) and not folder.startswith('.'):
            cases.append(folder_path)
    
    if not cases:
        print("No simulation cases found")
        return None, None
    
    print(f"\nFound {len(cases)} case(s)")
    print(f"Grid size: {grid_size}")
    
    for case_path in cases:
        try:
            extract_single_case(case_path, processed_dir, grid_size)
        except Exception as e:
            print(f"  Error: {e}")
    
    # Build combined dataset
    X, y = build_combined_dataset(processed_dir)
    
    return X, y


if __name__ == "__main__":
    X, y = extract_dataset("data/raw", "data/processed", grid_size=128)
    if X is not None:
        print(f"\nFinal shape: X={X.shape}, y={y.shape}")
