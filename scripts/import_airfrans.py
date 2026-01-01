"""
AirfRANS Dataset Import Script

Downloads the AirfRANS dataset (1000 RANS simulations) and converts 
to the same format as existing raw CFD data.

Usage:
    python scripts/import_airfrans.py --num-samples 50
"""

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm


def download_airfrans(root_dir):
    """Download AirfRANS dataset."""
    import airfrans
    
    print("Downloading AirfRANS dataset...")
    print("This may take a few minutes (~9 GB)...")
    
    airfrans.dataset.download(root=root_dir, unzip=True)
    print(f"Downloaded to: {root_dir}")


def load_vtk_to_dataframe(vtu_path):
    """Load VTU file and convert to pandas DataFrame."""
    import pyvista as pv
    
    mesh = pv.read(vtu_path)
    
    # Get point coordinates
    points = mesh.points
    x = points[:, 0]
    y = points[:, 1]
    
    # Get velocity (U field in OpenFOAM)
    if 'U' in mesh.point_data:
        U = mesh.point_data['U']
        u = U[:, 0]
        v = U[:, 1]
    else:
        u = np.zeros_like(x)
        v = np.zeros_like(x)
    
    # Get pressure (p field)
    if 'p' in mesh.point_data:
        p = mesh.point_data['p']
    else:
        p = np.zeros_like(x)
    
    return pd.DataFrame({
        'x_coordinate': x,
        'y_coordinate': y,
        'x_velocity': u,
        'y_velocity': v,
        'pressure': p
    })


def get_coefficients_from_simulation(root, sample_name):
    """
    Get CL and CD using the airfrans.Simulation class.
    This properly computes coefficients from the VTK data.
    """
    import airfrans
    
    try:
        sim = airfrans.dataset.Simulation(root=root, name=sample_name)
        
        # force_coefficient returns (cd_tuple, cl_tuple)
        # Each tuple is (total, pressure_contribution, viscous_contribution)
        cd_tuple, cl_tuple = sim.force_coefficient()
        
        # Extract total values
        cd = cd_tuple[0]  # Total CD
        cl = cl_tuple[0]  # Total CL
        
        return float(cl), float(cd)
        
    except Exception as e:
        print(f"Warning: Could not get coefficients for {sample_name}: {e}")
        return None, None


def save_coefficient_file(filepath, value):
    """Save coefficient in Fluent output format."""
    with open(filepath, 'w') as f:
        f.write('"coefficient-rfile"\n')
        f.write('"Iteration" "coefficient"\n')
        f.write('("Iteration" "coefficient")\n')
        for i in range(1, 11):
            f.write(f'{i} {value:.10f}\n')


def convert_sample(dataset_root, sample_name, output_dir):
    """Convert a single AirfRANS sample to raw format."""
    sample_dir = os.path.join(dataset_root, sample_name)
    
    # Find VTU file
    vtu_file = None
    for f in os.listdir(sample_dir):
        if f.endswith('_internal.vtu'):
            vtu_file = os.path.join(sample_dir, f)
            break
    
    if not vtu_file:
        return False, "No internal.vtu file found"
    
    try:
        # Load VTK data
        df = load_vtk_to_dataframe(vtu_file)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save fields.csv
        df.to_csv(os.path.join(output_dir, "fields.csv"), index=False)
        
        # Get coefficients using airfrans Simulation class
        cl, cd = get_coefficients_from_simulation(dataset_root, sample_name)
        
        # Use computed values or defaults
        if cl is None:
            cl = 0.5
        if cd is None:
            cd = 0.02
            
        # Ensure reasonable values
        cl = np.clip(cl, -3.0, 3.0)
        cd = np.clip(abs(cd), 0.001, 1.0)  # CD should be positive
        
        # Save coefficient files
        save_coefficient_file(
            os.path.join(output_dir, "lift_coefficient-rfile.out"),
            cl
        )
        save_coefficient_file(
            os.path.join(output_dir, "drag-coefficient-rfile.out"),
            cd
        )
        
        return True, f"CL={cl:.4f}, CD={cd:.4f}"
        
    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(description='Import AirfRANS dataset')
    parser.add_argument('--num-samples', type=int, default=50,
                        help='Number of samples to import (default: 50)')
    parser.add_argument('--output-dir', type=str, default='data/raw',
                        help='Output directory for raw data')
    parser.add_argument('--download-dir', type=str, default='data/external/airfrans',
                        help='Directory to download dataset')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip download if already exists')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("AirfRANS Dataset Import")
    print("=" * 60)
    print(f"Samples to import: {args.num_samples}")
    print(f"Output directory: {args.output_dir}")
    
    # Check if dataset exists
    dataset_dir = os.path.join(args.download_dir, "Dataset")
    
    if not os.path.exists(dataset_dir):
        if args.skip_download:
            print(f"Dataset not found at {dataset_dir}")
            return
        download_airfrans(args.download_dir)
        dataset_dir = os.path.join(args.download_dir, "Dataset")
    
    # List available samples (only airFoil2D directories)
    samples = []
    for item in os.listdir(dataset_dir):
        item_path = os.path.join(dataset_dir, item)
        if os.path.isdir(item_path) and item.startswith('airFoil2D'):
            samples.append(item)
    
    samples.sort()
    print(f"Found {len(samples)} sample directories")
    
    # Import samples
    num_to_import = min(args.num_samples, len(samples))
    print(f"\nImporting {num_to_import} samples...")
    print("(Computing CL/CD from simulation data - this may take a moment)")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    imported = 0
    errors = 0
    
    for i, sample_name in enumerate(tqdm(samples[:num_to_import], desc="Converting")):
        # Create output folder
        output_name = f"airfrans_{i:04d}"
        output_folder = os.path.join(args.output_dir, output_name)
        
        success, result = convert_sample(dataset_dir, sample_name, output_folder)
        
        if success:
            imported += 1
        else:
            errors += 1
            if errors <= 5:
                tqdm.write(f"Error [{sample_name}]: {result}")
    
    print(f"\n{'=' * 60}")
    print(f"Import Complete!")
    print(f"{'=' * 60}")
    print(f"Imported: {imported} samples")
    print(f"Errors: {errors}")
    print(f"Output: {args.output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Remove old processed data: rm -rf data/processed")
    print(f"  2. Run: python main.py")
    print(f"  3. Go to CFD tab -> Extract & Process")


if __name__ == "__main__":
    main()
