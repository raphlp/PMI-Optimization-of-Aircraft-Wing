import os
import glob
import h5py
import numpy as np
import pandas as pd

def extract_dataset(data_dir, assoc_dir, output_dir):
    print(f"\n📂 Extraction des fichiers dans : {data_dir}")

    fields_to_extract = ["SV_P", "SV_U", "SV_V", "SV_DENSITY"]

    # Lecture des fichiers CL/CD
    lift_df = pd.read_csv(os.path.join(assoc_dir, "lift_coefficient.tsv"), sep="\t")
    drag_df = pd.read_csv(os.path.join(assoc_dir, "drag_coefficient.tsv"), sep="\t")
    df_cd = pd.merge(lift_df, drag_df, on="case", suffixes=("_CL", "_CD"))

    dataset_X, dataset_y = [], []

    for h5_path in glob.glob(os.path.join(data_dir, "*.dat.h5")):
        case_name = os.path.splitext(os.path.basename(h5_path))[0]
        print(f"🔍 Lecture du fichier : {case_name}")

        with h5py.File(h5_path, "r") as f:
            case_inputs = []
            for field in fields_to_extract:
                path = f"results/1/phase-1/cells/{field}/1"
                if path in f:
                    arr = np.array(f[path])
                    case_inputs.append(arr)
                else:
                    print(f"⚠️ Champ {field} manquant dans {case_name}")
            X_case = np.stack(case_inputs, axis=-1)
            dataset_X.append(X_case)

        row = df_cd[df_cd["case"] == case_name]
        if not row.empty:
            CL = row.iloc[0]["value_CL"]
            CD = row.iloc[0]["value_CD"]
            dataset_y.append([CL, CD])
        else:
            print(f"⚠️ CL/CD manquants pour {case_name}")

    X = np.array(dataset_X, dtype=np.float32)
    y = np.array(dataset_y, dtype=np.float32)

    np.save(os.path.join(output_dir, "CFD_X.npy"), X)
    np.save(os.path.join(output_dir, "CFD_y.npy"), y)

    print("\n✅ Extraction terminée.")
    print(f"  → X shape: {X.shape}")
    print(f"  → y shape: {y.shape}")
    print(f"💾 Fichiers sauvegardés dans : {output_dir}")
