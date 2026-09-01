"""Central privacy: Thresholded Central-DP Lasso against Cai, Wang and Zhang (2021).

Both procedures see the same clipped statistics, the same total budget and the same
iteration count, so the comparison isolates how the budget is spent. We perturb the
whole gradient once per round; the comparator spends its per-round budget inside a
private top-k selection.

The scales that matter are the per-record gradient sensitivities. Writing
`D2 = R(RC+B)/n` for the l2 sensitivity and `Dinf` for the l-infinity one, our noise
contributes about `sqrt(s) D2 / eps` and the peeling mechanism about
`sqrt(s) s Dinf / eps`, so the ratio is `s Dinf / D2`. On a design whose row norm is
spread across the coordinates, `Dinf ~ D2 / sqrt(p)`, which predicts that we lose when
`s < sqrt(p)` and win when `s > sqrt(p)`. This sweep tests that prediction.
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

from dp_primitives import hard_threshold, peeling, project_l2, soft_threshold
from dp_primitives import central_trajectory_noise_scale

PROJECTION_RADIUS = 1.0
FEATURE_RADIUS = 2.0
LABEL_BOUND = 2.6
ITERATIONS = 8

MODES = {
    "pilot": dict(seeds=[8401, 8402], sizes=[50_000, 200_000], dims=[100, 200, 500],
                  sparsities=[5, 10, 20, 40], epsilons=[0.95]),
    # n = 50,000 leaves both arms at the zero estimator, so the grid starts above it.
    "confirm": dict(seeds=[8501, 8502, 8503, 8504, 8505, 8506, 8507, 8508],
                    sizes=[100_000, 200_000, 400_000], dims=[100],
                    sparsities=[5, 10, 20, 40], epsilons=[0.95]),
}


def make_data(rng, n, p, s_star):
    x = rng.standard_normal((n, p), dtype=np.float32)
    beta = np.zeros(p, dtype=np.float32)
    support = rng.choice(p, size=s_star, replace=False)
    beta[support] = np.arange(1, s_star + 1, dtype=np.float32) ** -1.5
    beta /= np.linalg.norm(beta)
    y = x @ beta + np.float32(0.5) * rng.standard_normal(n, dtype=np.float32)
    scale = np.minimum(1.0, FEATURE_RADIUS / np.maximum(np.linalg.norm(x, axis=1), 1e-12))
    return (x * scale[:, None].astype(np.float32),
            np.clip(y, -LABEL_BOUND, LABEL_BOUND), beta)


def smoothness(x, rng):
    n, p = x.shape
    v = rng.standard_normal(p).astype(np.float32)
    v /= np.linalg.norm(v)
    for _ in range(25):
        w = (x.T @ (x @ v)) / np.float32(n)
        v = w / max(float(np.linalg.norm(w)), 1e-12)
    return max(float(v @ ((x.T @ (x @ v)) / np.float32(n))), 1e-8)


def gradient(x, y, beta, n):
    return np.asarray(x.T @ (x @ beta.astype(np.float32) - y), dtype=np.float64) / n


def run_cell(seed, n, p, s_star, epsilon):
    rng = np.random.default_rng([seed, n, p, s_star, int(epsilon * 1000)])
    x, y, beta_star = make_data(rng, n, p, s_star)
    smooth = smoothness(x, rng)
    step = 1.0 / smooth
    delta = 10.0 / n**1.1
    lam = 0.75 * math.sqrt(2.0 * math.log(p) / n)
    sensitivity_l2 = 2.0 * FEATURE_RADIUS * (FEATURE_RADIUS * PROJECTION_RADIUS
                                             + LABEL_BOUND) / n
    grad_sd = central_trajectory_noise_scale(
        n=n, iterations=ITERATIONS, feature_radius=FEATURE_RADIUS,
        label_bound=LABEL_BOUND, projection_radius=PROJECTION_RADIUS,
        epsilon=epsilon, delta=delta)

    ours = np.zeros(p)
    for _ in range(ITERATIONS):
        noisy = gradient(x, y, ours, n) + rng.normal(0.0, grad_sd, size=p)
        ours = project_l2(hard_threshold(soft_threshold(ours - step * noisy,
                                                        step * lam), s_star),
                          PROJECTION_RADIUS)

    cai = np.zeros(p)
    for _ in range(ITERATIONS):
        half = cai - step * gradient(x, y, cai, n)
        cai = project_l2(peeling(half, sparsity=s_star,
                                 epsilon=epsilon / ITERATIONS,
                                 delta=delta / ITERATIONS,
                                 sensitivity_linf=step * sensitivity_l2, rng=rng),
                         PROJECTION_RADIUS)

    plain = np.zeros(p)
    for _ in range(150):
        plain = project_l2(soft_threshold(plain - step * gradient(x, y, plain, n),
                                          step * lam), PROJECTION_RADIUS)

    common = dict(seed=seed, n=n, p=p, s_star=s_star, epsilon=epsilon, delta=delta,
                  grad_sd=grad_sd, sensitivity_l2=sensitivity_l2,
                  ratio_prediction=s_star / math.sqrt(p))
    return [{**common, "method": name,
             "l2_error": float(np.linalg.norm(beta - beta_star)),
             "support": int(np.count_nonzero(beta))}
            for name, beta in (("Thresholded Central-DP Lasso (ours)", ours),
                               ("Cai et al. DP-IHT", cai),
                               ("Non-private", plain),
                               ("Zero estimator", np.zeros(p)))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=sorted(MODES), default="pilot")
    args = parser.parse_args()
    spec = MODES[args.mode]
    started = time.perf_counter()

    rows = []
    for seed in spec["seeds"]:
        for n in spec["sizes"]:
            for p in spec["dims"]:
                for s_star in spec["sparsities"]:
                    for epsilon in spec["epsilons"]:
                        rows += run_cell(seed, n, p, s_star, epsilon)
            print(f"  seed {seed} n={n} at {time.perf_counter() - started:.0f}s", flush=True)

    out = Path(__file__).resolve().parent / "results" / f"central_{args.mode}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    out.with_suffix(".json").write_text(json.dumps({
        "phase": "confirmation" if args.mode == "confirm" else "exploration",
        "mode": args.mode, **spec, "iterations": ITERATIONS,
        "feature_radius": FEATURE_RADIUS, "label_bound": LABEL_BOUND,
        "projection_radius": PROJECTION_RADIUS, "delta_rule": "10/n^1.1",
        "matching": "same clipped statistics, same total budget, same iteration count",
    }, indent=2), encoding="utf-8")
    print(f"wrote {out} in {(time.perf_counter() - started)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
