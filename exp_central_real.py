"""Central privacy on ACS records: our procedure against Cai, Wang and Zhang (2021).

Central privacy needs `n` of order `3e4` before its gradient noise falls below the
signal, and it is insensitive to the ambient dimension over the range considered, so the
ACS income design at `p = 105` with `n` in the hundreds of thousands sits inside its
usable regime. That is the same design used for the full-record case, reused here.

Rows are normalized and responses are +-1, so the clipping constants are exact rather
than assumed: the feature radius is 1 and the response bound is 1, giving a per-record
gradient sensitivity of `2 R (R C + B) / n = 4 / n`.

Both arms see the same statistics, the same total budget and the same iteration count.
The threshold level is swept, and the comparator is credited with its best cell.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

from dp_primitives import peeling
from dp_primitives import central_trajectory_noise_scale
from exp_empirical_fullrecord import (TEST_SIZE, FEATURES, hard, load_public, nmse,
                                      one_hot, project, smoothness, soft)

FEATURE_RADIUS = 1.0        # exact: rows lie on the unit sphere
LABEL_BOUND = 1.0           # exact: responses are +-1
PROJECTION_RADIUS = 1.0
ITERATIONS = 8

MODES = {
    "pilot": dict(seeds=[8601, 8602], sizes=[100_000, 400_000], epsilons=[0.95],
                  levels=[5, 20, 50, 100], lambdas=[0.1, 0.3, 1.0]),
    "confirm": dict(seeds=[8701, 8702, 8703, 8704, 8705, 8706, 8707, 8708],
                    sizes=[25_000, 50_000, 100_000, 200_000], epsilons=[0.95],
                    levels=[5, 20, 50, 100], lambdas=[None]),
}


def run_seed(seed, spec, public, labels):
    largest = max(spec["sizes"])
    rng = np.random.default_rng(seed + 600_000)
    index = rng.choice(len(public), size=largest + TEST_SIZE, replace=False)
    features = public.iloc[index][FEATURES].to_numpy()
    train_x, test_x = one_hot(features[:largest], features[largest:])
    train_y, test_y = labels[index][:largest], labels[index][largest:]
    p = train_x.shape[1]
    rows = []

    for n in spec["sizes"]:
        x, y = train_x[:n], train_y[:n]
        mean_n = float(y.mean())
        smooth = smoothness(x, seed=seed + 900_000 + n)
        step = 1.0 / smooth
        delta = 10.0 / n**1.1
        sensitivity = 2.0 * FEATURE_RADIUS * (FEATURE_RADIUS * PROJECTION_RADIUS
                                              + LABEL_BOUND) / n
        common = dict(seed=seed, n=n, p=p, delta=delta, smoothness=smooth,
                      sensitivity=sensitivity, test_size=len(test_y))

        def grad(beta):
            return np.asarray(x.T @ (x @ beta - y)).ravel() / n

        plain_level = math.sqrt(2.0 * math.log(p) / n)
        for level in spec["levels"]:
            for multiplier in spec["lambdas"]:
                plain = np.zeros(p)
                for _ in range(ITERATIONS):
                    plain = project(hard(soft(plain - step * grad(plain),
                                              step * multiplier * plain_level), level))
                rows.append({**common, "epsilon": math.nan,
                             "method": "Non-private, same procedure",
                             "level": level, "multiplier": multiplier,
                             "grad_sd": 0.0,
                             "nmse": nmse(test_x, test_y, plain, mean_n),
                             "support": int(np.count_nonzero(plain))})
        plain = np.zeros(p)
        for _ in range(150):
            plain = project(soft(plain - step * grad(plain), step * plain_level))
        rows.append({**common, "epsilon": math.nan, "method": "Non-private proximal",
                     "level": math.nan, "multiplier": math.nan, "grad_sd": math.nan,
                     "nmse": nmse(test_x, test_y, plain, mean_n),
                     "support": int(np.count_nonzero(plain))})
        rows.append({**common, "epsilon": math.nan, "method": "Constant predictor",
                     "level": math.nan, "multiplier": math.nan, "grad_sd": math.nan,
                     "nmse": nmse(test_x, test_y, np.zeros(p), mean_n), "support": 0})

        for epsilon in spec["epsilons"]:
            grad_sd = central_trajectory_noise_scale(
                n=n, iterations=ITERATIONS, feature_radius=FEATURE_RADIUS,
                label_bound=LABEL_BOUND, projection_radius=PROJECTION_RADIUS,
                epsilon=epsilon, delta=delta)
            for level in spec["levels"]:
                for multiplier in spec["lambdas"]:
                    beta = np.zeros(p)
                    inner = np.random.default_rng([seed, n, level, int(multiplier * 100)])
                    for _ in range(ITERATIONS):
                        noisy = grad(beta) + inner.normal(0.0, grad_sd, size=p)
                        beta = project(hard(soft(beta - step * noisy,
                                                 step * multiplier * plain_level), level))
                    rows.append({**common, "epsilon": epsilon,
                                 "method": "Thresholded Central-DP Lasso (ours)",
                                 "level": level, "multiplier": multiplier,
                                 "grad_sd": grad_sd,
                                 "nmse": nmse(test_x, test_y, beta, mean_n),
                                 "support": int(np.count_nonzero(beta))})
                beta = np.zeros(p)
                inner = np.random.default_rng([seed, n, level, 999])
                for _ in range(ITERATIONS):
                    half = beta - step * grad(beta)
                    beta = project(peeling(half, sparsity=level,
                                           epsilon=epsilon / ITERATIONS,
                                           delta=delta / ITERATIONS,
                                           sensitivity_linf=step * sensitivity,
                                           rng=inner))
                rows.append({**common, "epsilon": epsilon, "method": "Cai et al. DP-IHT",
                             "level": level, "multiplier": math.nan, "grad_sd": grad_sd,
                             "nmse": nmse(test_x, test_y, beta, mean_n),
                             "support": int(np.count_nonzero(beta))})
        print(f"  seed={seed} n={n} p={p} grad_sd={grad_sd:.5f}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=sorted(MODES), default="pilot")
    parser.add_argument("--multiplier", type=float, default=None)
    args = parser.parse_args()
    spec = dict(MODES[args.mode])
    if spec["lambdas"] == [None]:
        if args.multiplier is None:
            parser.error("confirm mode needs --multiplier")
        spec["lambdas"] = [args.multiplier]

    started = time.perf_counter()
    public = load_public()
    labels = np.where(public.PINCP.to_numpy() > 50_000, 1.0, -1.0)
    print(f"loaded {len(public):,} rows in {time.perf_counter()-started:.0f}s", flush=True)

    rows = []
    for seed in spec["seeds"]:
        rows += run_seed(seed, spec, public, labels)

    out = Path(__file__).resolve().parent / "results" / f"central_real_{args.mode}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    out.with_suffix(".json").write_text(json.dumps({
        "phase": "confirmation" if args.mode == "confirm" else "exploration",
        "mode": args.mode, **spec, "iterations": ITERATIONS,
        "feature_radius": FEATURE_RADIUS, "label_bound": LABEL_BOUND,
        "clipping_note": "both constants are exact on this design, not assumed",
        "task": "ACS income above 50,000, CA/NY/TX 2018 1-Year",
        "metric": "test NMSE against the training-mean predictor",
    }, indent=2), encoding="utf-8")
    print(f"wrote {out} in {(time.perf_counter()-started)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
