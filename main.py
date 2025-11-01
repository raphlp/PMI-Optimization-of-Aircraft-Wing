import os
from src.extract_data import extract_dataset

def main():
    print("\n🚀 Projet PMI – Optimisation robuste d’aile (CFD + CNN)")
    print("=====================================================")
    print("1️⃣  Extraire les données CFD → Dataset")
    print("2️⃣  Quitter")
    choice = input("\nChoisis une option : ")

    if choice == "1":
        data_dir = os.path.join("data", "raw")
        output_dir = os.path.join("data", "processed")
        assoc_dir = os.path.join("data", "associations")
        os.makedirs(output_dir, exist_ok=True)
        extract_dataset(data_dir, assoc_dir, output_dir)
    else:
        print("👋 Fin du programme.")

if __name__ == "__main__":
    main()
