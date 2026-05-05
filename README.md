# Reproducibility Code: Intrinsic-Dimension Empirical Bernstein Inequalities

This repository contains the Python scripts to reproduce the empirical simulations for the NeurIPS submission: **"Intrinsic-dimension empirical Bernstein inequalities for bounded self-adjoint operators."**

The code is divided into two distinct experiments:
1. `meb`: Evaluates the Operator Empirical Bernstein (OEB) bound against state-of-the-art ambient bounds under varying spectral geometries. `run_meb.py` generates the data, `read_meb.py` generates the plots.
2. `run_kpca_simulations.py`: Demonstrates the application of OEB to infinite-dimensional Reproducing Kernel Hilbert Spaces (RKHS) via Kernel PCA. `run_distributions_kpca.py` generates the data, `read_distributions_kpca.py` generates the plots.

## Requirements

The simulations are written in standard Python and rely on common scientific computing libraries. To install the required dependencies, run:

```bash
pip install numpy pandas matplotlib scipy tqdm