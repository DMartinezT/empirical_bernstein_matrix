#!/usr/bin/env python3
"""Reconstruct Reviewer 1 experiments; outputs always come from fresh computation."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

# Avoid nested BLAS parallelism, including Apple's Accelerate (not detected by
# threadpoolctl). Each --jobs worker uses one numerical-library thread.
for _name in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import scipy
from threadpoolctl import threadpool_limits, threadpool_info

from bounds import (bernstein_radius, commutator_counts, exact_population,
                    meb1_radius, meb2_radius, oeb_radius, operator_norm,
                    paired_statistics)

METHODS = ("oeb", "meb1", "meb2")
ROOT = Path(__file__).resolve().parent
DEFAULT_RANKS = {"noncommuting": [3, 5, 10, 20], "rkhs": [32, 64, 128, 256]}


def random_signs(seed, stream, trial, rank, n, condition=0):
    # Coordinate-major layout means adding larger truncations preserves prefixes.
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(
        [seed, stream, condition, trial])))
    return (2*rng.integers(0, 2, size=(rank, n), dtype=np.int8)-1).T.copy()


def run_trial(task):
    experiment, trial, n, ranks, seed, delta, solver = task
    rows = []
    with threadpool_limits(limits=1):
        if experiment == "rkhs":
            shared = random_signs(seed, 1, trial, max(ranks), n)
        for rank in ranks:
            if experiment == "noncommuting":
                signs = random_signs(seed, 0, trial, rank, n, condition=rank)
                weights = np.full(rank, 1/rank)
                dimension = 1000
            else:
                signs = shared[:, :rank]
                weights = 0.1*0.9**np.arange(rank)
                dimension = rank
            features = signs*np.sqrt(weights)
            c, variance, trace, effective_rank = exact_population(weights)
            stats = paired_statistics(features)
            oracle = bernstein_radius(variance, effective_rank, n, delta, c)
            ambient = bernstein_radius(variance, dimension, n, delta, c)
            oeb = oeb_radius(stats, n, delta, c, c*c)
            meb1 = meb1_radius(oeb["variance_hat"], n, dimension, delta, c)
            meb2, center = meb2_radius(features, dimension, delta, c, solver)
            mean = features.T @ features/n
            error = operator_norm(mean-np.diag(weights))
            weighted_error = operator_norm(center-np.diag(weights))
            row = {"experiment": experiment, "rank": rank, "dimension": dimension,
                   "n": n, "trial": trial, "master_seed": seed,
                   "sample_sha256": hashlib.sha256(signs.tobytes()).hexdigest(),
                   "delta": delta, "c": c, "B": c*c,
                   "variance_exact": variance, "trace_exact": trace,
                   "effective_rank_exact": effective_rank, "intrinsic_oracle": oracle,
                   "ambient_oracle": ambient, "ambient_ratio": ambient/oracle,
                   "sample_mean_error": error, "meb2_weighted_mean_error": weighted_error,
                   **oeb, "meb1_radius": meb1, "meb2_radius": meb2}
            for method in METHODS:
                row[f"{method}_ratio"] = row[f"{method}_radius"]/oracle
                method_error = weighted_error if method == "meb2" else error
                row[f"{method}_covered"] = int(method_error <= row[f"{method}_radius"])
            if experiment == "noncommuting":
                count, pairs, theory = commutator_counts(signs)
                row.update(commutator_nonzero_count=count, commutator_pair_count=pairs,
                           commutator_frequency=count/pairs, commutator_probability_exact=theory)
            else:
                row.update(commutator_nonzero_count="", commutator_pair_count="",
                           commutator_frequency="", commutator_probability_exact="")
            rows.append(row)
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    from scipy.stats import beta
    results = []
    keys = sorted({(r["experiment"], r["rank"]) for r in rows})
    for experiment, rank in keys:
        group = [r for r in rows if (r["experiment"], r["rank"]) == (experiment, rank)]
        k = len(group)
        for method in METHODS:
            ratios = np.array([r[f"{method}_ratio"] for r in group])
            hits = sum(r[f"{method}_covered"] for r in group)
            sd = float(ratios.std(ddof=1)) if k > 1 else float("nan")
            results.append({"experiment": experiment, "rank": rank, "method": method,
                            "n": group[0]["n"], "trials": k,
                            "mean_ratio": float(ratios.mean()), "sd_ratio": sd,
                            "mc_standard_error": sd/np.sqrt(k),
                            "q025_ratio": float(np.quantile(ratios, .025)),
                            "q975_ratio": float(np.quantile(ratios, .975)),
                            "mean_radius": float(np.mean([r[f"{method}_radius"] for r in group])),
                            "intrinsic_oracle": group[0]["intrinsic_oracle"],
                            "effective_rank_exact": group[0]["effective_rank_exact"],
                            "covered_trials": hits, "coverage": hits/k,
                            "coverage_cp95_lower": float(beta.ppf(.025, hits, k-hits+1)) if hits else 0.,
                            "coverage_cp95_upper": float(beta.ppf(.975, hits+1, k-hits)) if hits < k else 1.,
                            "commutator_frequency": (sum(r["commutator_nonzero_count"] for r in group)
                                /sum(r["commutator_pair_count"] for r in group)) if experiment == "noncommuting" else "",
                            "commutator_probability_exact": group[0]["commutator_probability_exact"]})
    return results


def reference_comparison(summary):
    with (ROOT/"rebuttal_reference.csv").open() as f:
        refs = {(r["experiment"], int(r["rank"]), r["method"]): r for r in csv.DictReader(f)}
    result = []
    for row in summary:
        key = row["experiment"], row["rank"], row["method"]
        if key not in refs:
            continue
        ref = refs[key]
        target = float(ref["reported_mean_ratio"])
        same_size = row["n"] == int(ref["n"]) and row["trials"] == int(ref["trials"])
        result.append({"experiment": key[0], "rank": key[1], "method": key[2],
                       "reported_mean_ratio": target, "computed_mean_ratio": row["mean_ratio"],
                       "difference": row["mean_ratio"]-target,
                       "same_n_and_trial_count": int(same_size),
                       "matches_reported_3_decimals": int(round(row["mean_ratio"], 3) == target)})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=["both", "noncommuting", "rkhs"], default="both")
    parser.add_argument("--output", type=Path, default=ROOT/"results")
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--delta", type=float, default=.05)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--solver", choices=["iterative", "dense"], default="iterative")
    parser.add_argument("--trials", type=int, help="Override trial counts (40 and 100 by default)")
    parser.add_argument("--n", type=int, help="Override sample sizes (20000 and 10000 by default)")
    parser.add_argument("--smoke", action="store_true", help="Use n=256, two trials per experiment")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.seed < 0 or args.jobs < 1 or not 0 < args.delta < 1:
        parser.error("seed >= 0, jobs >= 1 and 0 < delta < 1 required")
    if args.n is not None and (args.n < 8 or args.n % 4):
        parser.error("n must be a multiple of four and at least eight")
    if args.trials is not None and args.trials < 1:
        parser.error("trials must be positive")
    if args.smoke and (args.n is not None or args.trials is not None):
        parser.error("--smoke cannot be combined with --n or --trials")
    output = args.output.resolve()
    if (output/"manifest.json").exists() and not args.overwrite:
        parser.error("output already contains a run; use a new --output or --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    experiments = list(DEFAULT_RANKS) if args.experiment == "both" else [args.experiment]
    tasks = []
    for experiment in experiments:
        n = 256 if args.smoke else args.n or (20000 if experiment == "noncommuting" else 10000)
        k = 2 if args.smoke else args.trials or (40 if experiment == "noncommuting" else 100)
        tasks.extend((experiment, t, n, DEFAULT_RANKS[experiment], args.seed, args.delta, args.solver)
                     for t in range(k))
    if "noncommuting" in experiments:
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([args.seed, 2])))
        q, _ = np.linalg.qr(rng.standard_normal((1000, 20)))
        np.savez_compressed(output/"embedding.npz", Q=q)
    manifest = {"status": "running", "reconstruction_not_recovered_original_code": True,
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "command": sys.argv, "options": {**vars(args), "output": str(output)},
                "python": platform.python_version(), "numpy": np.__version__,
                "scipy": scipy.__version__, "platform": platform.platform(),
                "random_generator": "PCG64/SeedSequence", "threads_per_worker": 1,
                "threadpools": threadpool_info(),
                "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted(ROOT.glob("*.py"))},
                "conditions": [{"experiment": e, "ranks": DEFAULT_RANKS[e],
                                "n": next(t[2] for t in tasks if t[0] == e),
                                "trials": sum(t[0] == e for t in tasks)} for e in experiments]}
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    start = time.perf_counter()
    rows = []
    try:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(run_trial, task): task for task in tasks}
            for done, future in enumerate(as_completed(futures), 1):
                rows.extend(future.result())
                task = futures[future]
                rows.sort(key=lambda r: (r["experiment"], r["rank"], r["trial"]))
                write_csv(output/"trials.csv", rows)
                print(f"{done}/{len(tasks)} trials: {task[0]} #{task[1]} "
                      f"({time.perf_counter()-start:.1f}s)", flush=True)
        summary = summarize(rows)
        write_csv(output/"summary.csv", summary)
        write_csv(output/"comparison_to_rebuttal.csv", reference_comparison(summary))
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest["status"] = "incomplete"
        manifest["error"] = repr(exc)
        raise
    finally:
        manifest["elapsed_seconds"] = time.perf_counter()-start
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["completed_condition_trials"] = len(rows)
        manifest["output_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in output.iterdir() if p.suffix in (".csv", ".npz")}
        (output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    for row in summary:
        print(f'{row["experiment"]:14s} {row["rank"]:3d} {row["method"]:4s} '
              f'{row["mean_ratio"]:.6f} (MC SE {row["mc_standard_error"]:.6f})')


if __name__ == "__main__":
    main()
