# ✈️ Robust Wing Shape Optimization using CFD and Deep Learning

This project aims to **optimize the aerodynamic shape of an aircraft wing** under uncertainty by combining **Computational Fluid Dynamics (CFD)** and **Convolutional Neural Networks (CNNs)**.  
The approach accelerates aerodynamic performance prediction (lift and drag coefficients) while maintaining physical accuracy, enabling **robust and energy-efficient wing design**.

---

## 🧩 Project Overview

The project integrates:
1. **CFD Simulation (ANSYS Fluent)**  
   → Generates physical flow fields (pressure, velocity, density, etc.) around airfoils.  
2. **CNN-Based Surrogate Model**  
   → Learns to predict aerodynamic coefficients (*CL*, *CD*) directly from CFD field data.  
3. **Bayesian Optimization** (later phase)  
   → Incorporates uncertainty quantification to identify the most robust wing geometries.

---

## 📁 Repository Structure

```
pmi-wing-optimization/
│
├── data/
│   ├── raw/              # Raw CFD data (.h5) and coefficient files (.tsv)
│   ├── processed/        # Extracted datasets ready for ML (CFD_X.npy, CFD_y.npy)
│   └── associations/     # CL/CD correspondence files
│
├── src/
│   ├── extract_data.py   # CFD → dataset extraction logic
│   ├── utils_io.py       # Utility functions for I/O (optional)
│   └── visualize.py      # CFD field visualization tools (optional)
│
├── main.py               # CLI entry point for data extraction
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## ⚙️ Setup & Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/pmi-wing-optimization.git
cd pmi-wing-optimization
```

### 2️⃣ Create a virtual environment
```bash
conda create -n pmi python=3.11
conda activate pmi
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Run the main menu
```bash
python main.py
```

You’ll see:
```
🚀 PMI – Robust Wing Optimization (CFD + CNN)
1️⃣  Extract CFD → Dataset
2️⃣  Quit
```

The extraction script will:
- Load CFD `.h5` files from `data/raw/`
- Load lift & drag coefficients from `data/associations/`
- Build the dataset (`CFD_X.npy`, `CFD_y.npy`)
- Save it in `data/processed/`

---

## 📊 Output Data

- **`CFD_X.npy`** → CFD field tensors (pressure, velocity, density, etc.)  
- **`CFD_y.npy`** → Target aerodynamic coefficients `[CL, CD]`  

These arrays can be used directly for CNN training with TensorFlow or PyTorch.

---

## 🧠 Next Steps

- 🧩 Implement the CNN architecture (TensorFlow / PyTorch)
- 📈 Train on the generated dataset
- 🔁 Integrate uncertainty modeling (PINNs or Bayesian optimization)
- ⚙️ Automate CFD-to-ML pipelines

---

## 👥 Authors
**IPSA Master Project 2025–2026**  
**Title:** Robust Wing Shape Optimization using CFD and Deep Learning with Uncertainty  
Supervised by *Dr. Hammou El-Otmany* (IPSA Paris)  
Contributors: *CFD & Machine Learning team*  

---

## 📜 License
This project is released under the [MIT License](LICENSE).
