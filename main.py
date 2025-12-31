"""
PMI Project: Robust Wing Optimization using CFD and Deep Learning

Main entry point - launches the graphical user interface.
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def main():
    """Launch the PMI application."""
    try:
        from src.gui.app import run_app
        run_app()
    except ImportError as e:
        print(f"Error importing GUI: {e}")
        print("\nMake sure customtkinter is installed:")
        print("  pip install customtkinter")
        print("\nFalling back to console mode...")
        run_console()

def run_console():
    """Fallback console mode if GUI fails."""
    from src.extract_data import extract_dataset
    from src.train_cnn import train_cnn_model
    
    print("\n" + "="*60)
    print("  PMI Project: Robust Wing Optimization (CFD + CNN)")
    print("  Console Mode")
    print("="*60)
    print("\nOptions:")
    print("  1) Extract CFD data → Dataset")
    print("  2) Train CNN model")
    print("  3) Run full pipeline")
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
        extract_dataset(data_dir, output_dir, grid_size=128)
        train_cnn_model(output_dir, epochs=50)
    else:
        print("Exiting.")


if __name__ == "__main__":
    main()
