import os
import glob
import h5py
import numpy as np

def read_last_coefficient(path):
    """Reads the last coefficient (CL or CD) from a Fluent .out file"""
    with open(path, "r") as f:
        lines = f.readlines()
    values = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 2 and parts[0].isdigit():
            iteration = int(parts[0])
            value = float(parts[1])
            values.append((iteration, value))
    if not values:
        raise ValueError(f"No numeric data found in {path}")
    return values[-1][1]  # last value (final coefficient value)

def extract_dataset(data_dir, assoc_dir, output_dir):
    print(f"\nExtracting files from: {data_dir}")

    fields_to_extract = ["SV_P", "SV_U", "SV_V", "SV_DENSITY"]

    # Read final coefficients
    CL = read_last_coefficient(os.path.join(assoc_dir, "lift_coefficient-rfile.out"))
    CD = read_last_coefficient(os.path.join(assoc_dir, "drag_coefficient-rfile.out"))
    print(f"Coefficients read: CL = {CL:.4f}, CD = {CD:.4f}")

    dataset_X, dataset_y = [], []

    for h5_path in glob.glob(os.path.join(data_dir, "*.dat.h5")):
        case_name = os.path.splitext(os.path.basename(h5_path))[0]
        print(f"Reading file: {case_name}")

        with h5py.File(h5_path, "r") as f:
            case_inputs = []
            for field in fields_to_extract:
                path = f"results/1/phase-1/cells/{field}/1"
                if path in f:
                    arr = np.array(f[path])
                    case_inputs.append(arr)
                else:
                    print(f"Warning: Field {field} missing in {case_name}")

            X_case = np.stack(case_inputs, axis=-1)
            dataset_X.append(X_case)
            dataset_y.append([CL, CD])  # same coefficients for this case

    X = np.array(dataset_X, dtype=np.float32)
    y = np.array(dataset_y, dtype=np.float32)

    np.save(os.path.join(output_dir, "CFD_X.npy"), X)
    np.save(os.path.join(output_dir, "CFD_y.npy"), y)

    print("\nExtraction completed.")
    print(f"  → X shape: {X.shape}")
    print(f"  → y shape: {y.shape}")
    print(f"Files saved to: {output_dir}")
