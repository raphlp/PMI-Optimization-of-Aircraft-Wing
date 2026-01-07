# Data Directory

This directory contains the datasets used for training and evaluating the aerodynamic surrogate models.

## Directory Structure

```
data/
|-- processed/     # Pre-processed datasets ready for machine learning
|-- raw/           # Raw CFD simulation files (not included in repository)
|-- external/      # External datasets such as AirfRANS (not included)
```

---

## Downloading the Data

The processed datasets are hosted on Google Drive due to their size.

**[Download Processed Data](https://drive.google.com/drive/folders/1AxEvmxQJmi1tGZAJik45YjwulwIoOjQH?usp=sharing)**

After downloading, place the files in the `processed/` subdirectory.

---

## Processed Data

The `processed/` directory contains pre-processed numpy arrays ready for training:

| File | Description |
|------|-------------|
| `CFD_X.npy` | Input tensors: flow field data (pressure, velocity, density) interpolated on a uniform grid |
| `CFD_y.npy` | Target values: aerodynamic coefficients [CL, CD] |
| `*.keras` | Pre-trained model weights |

These files are generated from the raw CFD simulations using the extraction pipeline in `src/extract_data.py`.

---

## Raw Data (Available on Request)

The `raw/` directory is not included in the Google Drive download due to size constraints (~3 GB).

It contains the original ANSYS Fluent simulation exports in HDF5 format:
- Flow field variables (pressure, velocity components, density, turbulent viscosity)
- Mesh coordinates and geometry information
- Simulation metadata and convergence data

**If you need access to the raw CFD files** to reproduce the data extraction pipeline or for research purposes, please contact the authors:
- Raphael Laupies
- Noam Gomis
- Hugo Bensasson

---

## External Datasets

The `external/` directory may contain additional datasets used for extended training:

- **AirfRANS** (~25 GB): A large-scale CFD dataset for machine learning. Available at: https://github.com/Extrality/AirfRANS

These are not included in the repository and must be downloaded separately if needed.
