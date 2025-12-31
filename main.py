"""
PMI Project: Robust Wing Optimization using CFD and Deep Learning

Main entry point for the data extraction and CNN training pipeline.
"""

import os
from src.extract_data import extract_dataset
from src.train_cnn import train_cnn_model


def main():
    print("\n" + "="*60)
    print("  PMI Project: Robust Wing Optimization (CFD + CNN)")
    print("="*60)
    print("\nOptions:")
    print("  1) Extract CFD data → Dataset (interpolate to grid)")
    print("  2) Train CNN model")
    print("  3) Run full pipeline (extract + train)")
    print("  4) Exit")
    
    choice = input("\nSelect an option: ").strip()

    data_dir = os.path.join("data", "raw")
    output_dir = os.path.join("data", "processed")
    os.makedirs(output_dir, exist_ok=True)

    if choice == "1":
        extract_dataset(data_dir, output_dir, grid_size=128)

    elif choice == "2":
        train_cnn_model(output_dir, epochs=50)

    elif choice == "3":
        print("\n>>> Running full pipeline...")
        extract_dataset(data_dir, output_dir, grid_size=128)
        train_cnn_model(output_dir, epochs=50)

    else:
        print("Exiting program.")


if __name__ == "__main__":
    main()
