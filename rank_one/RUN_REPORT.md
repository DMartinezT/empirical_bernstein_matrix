# Completed reference run

These computed results with seed **20260919** are used in Tables 1 and 2 of the updated manuscript. Full means and Monte Carlo standard errors are recorded in [results/summary.csv](results/summary.csv).

The full run used 40 trials per noncommuting rank and 100 shared RKHS trials. All radii below are divided by the exact intrinsic oracle for their condition.

## Noncommuting updates: d=1000, n=20000

| Rank | OEB | MEB1 | MEB2 | Nonzero commutators |
|---:|---:|---:|---:|---:|
| 3 | 1.047485 | 1.664550 | 1.536921 | 74.904% |
| 5 | 1.075739 | 1.617280 | 1.466524 | 93.767% |
| 10 | 1.149251 | 1.587995 | 1.392311 | 75.135% |
| 20 | 1.280740 | 1.596624 | 1.339303 | 82.320% |

## RKHS truncations: n=10000

| Rank | OEB | MEB1 | MEB2 | Nonzero commutators |
|---:|---:|---:|---:|---:|
| 32 | 1.198252 | 1.329818 | 1.126595 | N/A |
| 64 | 1.204642 | 1.398580 | 1.179075 | N/A |
| 128 | 1.204869 | 1.465271 | 1.232202 | N/A |
| 256 | 1.204869 | 1.529912 | 1.283574 | N/A |

## Historical comparison and validation

The largest absolute difference from the rebuttal’s rounded ratios is **0.001689**; 17/24 entries round to the same three decimals. This comparison is historical; the current manuscript uses the computed results above. No seed search was performed. See [results/comparison_to_rebuttal.csv](results/comparison_to_rebuttal.csv) for all entries.

OEB uses the justified positive-semidefinite specialization with c, rather than the general self-adjoint 2c, in its final linear term. Both radii are exported, and the [README](README.md) explains the specialization used in the manuscript.

All 10 mathematical tests passed. The final output audit verified 560 unique condition-trials, 24 summaries, source/output hashes, exact analytic oracles, every radius normalization, coverage indicators, commutator counts, and summary means/standard deviations.

A full-size RKHS trial run with one worker agreed with the four-worker run at all four truncations (within 1e-12 relative tolerance).

Full simulation elapsed time: 19.93 minutes with four workers. Python 3.12.14, NumPy 2.3.5, SciPy 1.16.3.

Coverage is recorded separately per rank and method; shared RKHS ranks are not treated as independent trials. MEB2 uses its own weighted-mean center.
