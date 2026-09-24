# Reproducibility Code: Intrinsic-Dimension Empirical Bernstein Inequalities

Python code and recorded results for **“Intrinsic-dimension empirical Bernstein inequalities for bounded self-adjoint operators.”** The experiments compare the Operator Empirical Bernstein (OEB) radius with ambient-dimension bounds and examine its behavior in kernel settings.

## Experiments

| Experiment | Directory | Paper output |
| --- | --- | --- |
| Diagonal covariance matrices with isotropic, anisotropic, and polynomial spectra | [`meb/`](meb/) | Figure 1 |
| Gaussian-RBF kernel PCA under Gaussian, uniform, and exponential inputs | [`pca/`](pca/) | Figure 2 |
| Noncommuting rank-one updates in ambient dimension 1000 | [`rank_one/`](rank_one/) | Table 1 |
| Finite-rank truncations of a bounded Rademacher-feature RKHS | [`rank_one/`](rank_one/) | Table 2 |

The `pca/` directory also includes an additional comparison of RBF, Laplace, and inverse-multiquadric kernels. The two `rank_one/` experiments share a runner and bound implementation; this directory contains the `reviewer1_reproduction` package referenced in the manuscript. Their sampling laws, exact population moments, confidence allocation, boundedness constants, and formula-to-code correspondence are documented in [`rank_one/README.md`](rank_one/README.md).

```text
meb/                       Original covariance experiment, CSV, and plot
pca/                       Original kernel experiments and plotting scripts
  outputs/                 Recorded kernel results and plots
rank_one/
  bounds.py                OEB, MEB1, MEB2, and exact population oracles
  run_experiments.py       Noncommuting and RKHS truncation experiments
  tests/                   Mathematical checks against dense definitions
  results/                 Recorded full run used in Tables 1 and 2
  outputs/                 Local reruns (created on demand; ignored by Git)
  RUN_REPORT.md            Reference-run tables and validation details
  requirements.txt         Pinned dependencies for the new experiments
requirements.txt           Dependencies for all experiments and plotting
```

## Installation

Use Python 3.11 or newer. The recorded rank-one run used Python 3.12.14, NumPy 2.3.5, SciPy 1.16.3, and threadpoolctl 3.6.0. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`. No external datasets are required.

## New experiments: noncommuting updates and RKHS truncations

Run these commands from the repository root:

```bash
# Mathematical checks and a small end-to-end run.
python -m unittest discover -s rank_one/tests -v
python rank_one/run_experiments.py --smoke --jobs 2 --output rank_one/outputs/smoke

# Full experiment settings for both tables.
python rank_one/run_experiments.py --jobs 4 --output rank_one/outputs/full

# Either experiment separately.
python rank_one/run_experiments.py --experiment noncommuting --jobs 4 --output rank_one/outputs/noncommuting
python rank_one/run_experiments.py --experiment rkhs --jobs 4 --output rank_one/outputs/rkhs
```

The full-run defaults are:

| Setting | Noncommuting updates | RKHS truncations |
| --- | --- | --- |
| Sample size | 20,000 | 10,000 |
| Ranks | 3, 5, 10, 20 | 32, 64, 128, 256 |
| Monte Carlo trials | 40 per rank | 100, shared across truncations |
| Ambient dimension in MEB1/MEB2 | 1000 | Truncation rank |
| Failure probability | 0.05 | 0.05 |
| Master seed | 20260919 | 20260919 |
| Envelopes | `c = B = 1` | `c = 1 - 0.9**m`, `B = c**2` |

The RKHS weights are `p_j = 0.1 * 0.9**(j - 1)`; features are not renormalized after truncation. Every method uses the same observations for a given trial. OEB and MEB1 use the sample mean, while MEB2 uses its predictable-weight mean. The implementation retains the PSD specialization described in the paper.

The checked-in [`rank_one/results/summary.csv`](rank_one/results/summary.csv) contains the computed means and Monte Carlo standard errors used in the updated manuscript. [`trials.csv`](rank_one/results/trials.csv) contains all 560 condition-trials, and [`RUN_REPORT.md`](rank_one/RUN_REPORT.md) displays the reference tables. These are the current paper results. The separately labeled `rebuttal_reference.csv` and `results/comparison_to_rebuttal.csv` retain the earlier rounded values only for historical comparison.

Each new run writes trial and summary CSVs, a manifest with settings and hashes, and the noncommuting embedding when applicable. Existing run directories are protected; choose a new `--output` directory, or explicitly use `--overwrite` to replace a run. The default output is the checked-in `rank_one/results/`, so use the commands above to keep the reference run intact.

`--smoke` uses 256 observations and two trials per experiment, retaining all ranks. It checks the pipeline and does not reproduce the paper tables. `--n` and `--trials` can override the full settings; `n` must be divisible by four and at least eight. `--seed` selects another reproducible run, and `--solver dense` selects the reference eigensolver. Four workers took about 20 minutes for the recorded full run; runtime varies with hardware. Numerical libraries use one thread per worker.

## Original covariance and kernel experiments

These scripts use paths relative to the working directory. Run each command block in its indicated directory.

### Covariance matrices (Figure 1)

```bash
cd meb
python read_meb.py                 # Plot the recorded simulation_results.csv.
python run_meb.py                  # Regenerate the simulation CSV.
python read_meb.py                 # Plot the new results.
cd ..
```

The plot is `meb/neurips_simulation_plot.png`. Settings at the bottom of `run_meb.py` use `d = 3`, 50 trials, failure probability 0.05, and sample sizes 10,000 through 1,000,000. This script does not set a fixed random seed, so rerunning the simulation can change its Monte Carlo averages.

### Kernel PCA across input distributions (Figure 2)

```bash
cd pca
python read_distributions_kpca.py   # Plot the recorded CSV.
python run_distributions_kpca.py    # Regenerate outputs/kpca_distributions.csv.
python read_distributions_kpca.py
cd ..
```

The plot is `pca/outputs/kpca_distributions_plot.png`. The simulation uses an RBF kernel with `gamma = 0.1`, input dimension 10, seed 42, 20 trials, and sample sizes from 100 to 30,000. Its population oracle is estimated by Monte Carlo; the two new rank-one experiments use analytic population oracles. Kernel simulations form dense Gram matrices, so the largest sample sizes require substantial memory.

### Additional kernel comparison

```bash
cd pca
python read_kernels_kpca.py         # Plot the recorded kernel comparison.
python run_kernels_kpca.py          # Regenerate outputs/kpca_kernels_results.csv.
python read_kernels_kpca.py
cd ..
```

This produces `pca/outputs/kpca_kernels_plot.png`. The original simulations write to their existing CSV paths. To retain the supplied results, copy those CSVs before rerunning; plotting alone uses the recorded data.
