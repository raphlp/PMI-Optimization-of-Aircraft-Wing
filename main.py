import os
from src.extract_data import extract_dataset
from src.train_cnn import train_cnn_model

def main():
    print("\n--- PMI Project: Robust Wing Optimization (CFD + CNN) ---")
    print("1) Extract CFD data → Dataset")
    print("2) Train CNN model")
    print("3) Exit")
    choice = input("\nSelect an option: ")

    if choice == "1":
        data_dir = os.path.join("data", "raw")
        output_dir = os.path.join("data", "processed")
        assoc_dir = os.path.join("data", "associations")
        os.makedirs(output_dir, exist_ok=True)
        extract_dataset(data_dir, assoc_dir, output_dir)

    elif choice == "2":
        processed_dir = os.path.join("data", "processed")
        train_cnn_model(processed_dir)

    else:
        print("Exiting program.")

if __name__ == "__main__":
    main()
