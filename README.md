# Robust Wing Shape Optimization using CFD and Deep Learning

A comprehensive application for **optimizing aircraft wing aerodynamics** under uncertainty by combining **Computational Fluid Dynamics (CFD)** simulations with **Deep Learning** surrogate models. This project accelerates aerodynamic performance prediction (lift and drag coefficients) while maintaining physical accuracy, enabling robust and energy-efficient wing design.

---

## Authors

**IPSA Engineering School - Master Project 2024-2025**

- **Raphael Laupies**
- **Noam Gomis**
- **Hugo Bensasson**

Supervised by **Dr. Hammou El-Otmany** (IPSA Paris)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Repository Structure](#repository-structure)
4. [Requirements](#requirements)
5. [Installation](#installation)
6. [Data Download](#data-download)
7. [Usage](#usage)
8. [Application Workflow](#application-workflow)
9. [Technical Details](#technical-details)
10. [License](#license)

---

## Project Overview

Aircraft wing design optimization is a computationally expensive process that traditionally relies on numerous CFD simulations. This project addresses this challenge by developing a hybrid approach that:

1. **CFD Simulation (ANSYS Fluent)**: Generates high-fidelity flow fields (pressure, velocity, density, turbulence) around NACA airfoil geometries.

2. **Deep Learning Surrogate Models**: Trains Convolutional Neural Networks (CNN) to predict aerodynamic coefficients (CL, CD) directly from flow field data, reducing inference time from hours to milliseconds.

3. **Physics-Informed Neural Networks (PINNs)**: Incorporates physical constraints (Navier-Stokes equations) into the learning process for improved generalization.

4. **Bayesian Optimization**: Enables uncertainty quantification to identify robust wing geometries that perform well across varying conditions.

---

## Features

### Graphical User Interface
- Modern dark-themed interface built with CustomTkinter
- Two-step workflow: Field Generation and Prediction
- Real-time visualization of results and training progress

### Data Sources (Step 1)
- **CFD Panel**: Import and process ANSYS Fluent simulation data (.h5 format)
- **PINNs Panel**: Train Physics-Informed Neural Networks for flow field generation
- **LSTM Panel**: Time-series prediction for unsteady aerodynamics (coming soon)

### Prediction (Step 2)
- **CNN Prediction**: Load trained models to predict CL/CD coefficients
- Support for multiple trained models with comparison capabilities
- Visualization of predictions with uncertainty estimates

### Data Processing
- Automatic extraction of CFD field data from HDF5 files
- Grid interpolation for consistent tensor dimensions
- Coefficient association from TSV lookup tables

---

## Repository Structure

```
pmi-wing-optimization/
|
|-- data/                          # Data directory (download separately)
|   |-- raw/                       # Raw CFD simulation files (.h5)
|   |-- processed/                 # Processed ML-ready datasets (.npy)
|   |-- external/                  # External datasets (AirfRANS, etc.)
|
|-- src/
|   |-- gui/                       # Graphical User Interface
|   |   |-- app.py                 # Main application window
|   |   |-- theme.py               # UI theme configuration
|   |   |-- components/            # UI panels
|   |       |-- cfd_panel.py       # CFD data import panel
|   |       |-- pinns_panel.py     # PINNs training panel
|   |       |-- lstm_panel.py      # LSTM panel (placeholder)
|   |       |-- prediction_panel.py # CNN prediction panel
|   |
|   |-- models/                    # Model architectures
|   |   |-- naca_geometry.py       # NACA airfoil geometry utilities
|   |
|   |-- extract_data.py            # CFD to dataset extraction
|   |-- train_cnn.py               # CNN training script
|   |-- train_pinns.py             # PINNs training script
|   |-- bayesian_inference.py      # Bayesian optimization utilities
|
|-- scripts/
|   |-- import_airfrans.py         # AirfRANS dataset import script
|
|-- main.py                        # Application entry point
|-- requirements.txt               # Python dependencies
|-- .gitignore                     # Git ignore rules
|-- README.md                      # This file
```

---

## Requirements

### System Requirements
- Python 3.10 or higher
- 8 GB RAM minimum (16 GB recommended for training)
- GPU with CUDA support (optional, recommended for training)

### Python Dependencies
- numpy
- pandas
- scipy
- tensorflow (2.x)
- matplotlib
- customtkinter

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/raphlp/PMI-Optimization-of-Aircraft-Wing.git
cd PMI-Optimization-of-Aircraft-Wing
```

### 2. Create a Virtual Environment

Using Conda (recommended):
```bash
conda create -n pmi python=3.11
conda activate pmi
```

Or using venv:
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python main.py
```

The graphical interface should launch. If CustomTkinter is not available, a console fallback mode will be used.

---

## Data Download

The processed datasets are too large to be hosted on GitHub. Download them from the following Google Drive link:

**[Download Data (Google Drive)](https://drive.google.com/drive/folders/1AxEvmxQJmi1tGZAJik45YjwulwIoOjQH?usp=sharing)**

After downloading, place the contents into the `data/processed/` directory:

```
pmi-wing-optimization/
|-- data/
|   |-- processed/     # Place downloaded .npy files here
```

### Data Contents

| File | Description |
|------|-------------|
| `CFD_X.npy` | Flow field tensors (pressure, velocity, density) |
| `CFD_y.npy` | Target aerodynamic coefficients (CL, CD) |
| `*.keras` | Pre-trained model weights (if included) |

### Additional Data (On Request)

The following data is not included in the download due to size constraints:

- **Raw CFD files** (~3 GB): Original ANSYS Fluent simulation exports in HDF5 format. Contact the authors if you need to reproduce the data extraction pipeline.
- **AirfRANS dataset** (~25 GB): External dataset used for extended training. Available at [AirfRANS repository](https://github.com/Extrality/AirfRANS).

---

## Usage

### Launching the Application

```bash
python main.py
```

This launches the graphical user interface with the full feature set.

### Console Mode (Fallback)

If the GUI fails to load, a console mode is available:

```bash
python main.py
```

Console options:
1. Extract CFD data to dataset
2. Train CNN model
3. Run full pipeline
4. Exit

### Command-Line Scripts

#### Extract CFD Data
```bash
python -c "from src.extract_data import extract_dataset; extract_dataset('data/raw', 'data/processed', grid_size=128)"
```

#### Train CNN Model
```bash
python -c "from src.train_cnn import train_cnn_model; train_cnn_model('data/processed', epochs=50)"
```

#### Train PINNs Model
```bash
python -c "from src.train_pinns import train_pinns; train_pinns()"
```

---

## Application Workflow

### Step 1: Field Data Source

Choose one of the following methods to generate or import aerodynamic field data:

#### CFD Panel
1. Click on "CFD Simulation" in the sidebar
2. Select `.h5` files from `data/raw/`
3. Configure grid interpolation size (default: 128x128)
4. Click "Extract Dataset" to generate training data

#### PINNs Panel
1. Click on "PINNs" in the sidebar
2. Configure NACA airfoil parameters (4-digit code)
3. Set flow conditions (Mach number, angle of attack, Reynolds number)
4. Train the physics-informed network
5. Generate synthetic flow fields

### Step 2: Prediction

1. Click on "CNN Prediction" in the sidebar
2. Select a trained model from the dropdown
3. Choose input data source (CFD or PINNs generated)
4. View predicted CL/CD coefficients with uncertainty bounds
5. Compare predictions with ground truth (when available)

---

## Technical Details

### CNN Architecture

The surrogate model uses a convolutional neural network designed for regression:
- Input: Multi-channel flow field tensors (pressure, velocity components, density)
- Architecture: Conv2D layers with batch normalization and dropout
- Output: Two scalar values (CL, CD)

### PINNs Implementation

Physics-Informed Neural Networks incorporate the Navier-Stokes equations as soft constraints:
- Continuity equation residual
- Momentum equation residuals (x and y components)
- Boundary condition enforcement

### Data Format

CFD simulations are stored in HDF5 format with the following structure:
- Flow field variables: pressure, velocity (u, v), density, turbulent viscosity
- Geometry information: airfoil coordinates, mesh structure
- Metadata: simulation parameters, convergence data

---

## License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- IPSA Engineering School for academic support
- Dr. Hammou El-Otmany for project supervision
- ANSYS for CFD simulation software
- The TensorFlow and CustomTkinter development teams
