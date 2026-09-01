"""Label-local comparison in the submitted paper's layout.

Kept from the first submission: three data-generating settings, p = 2n, unit-norm
coefficients, l2 estimation error against the sample size, and the same baseline,
Label-LDP-IHT of Wang and Xu (2021).

Changed:

* the two arms consume the *same* released response vector, so the comparison isolates
  soft against hard thresholding rather than two calibration constants;
* the analytic Gaussian scale of Balle and Wang (2018) with delta = 10/n^1.1;
* the second setting is scaled to unit coordinate variance. As written in the submitted
  paper it draws X_ij ~ Unif[-1/sqrt(p), 1/sqrt(p)], so Var(x'beta*) = 1/(3p) and the
  signal vanishes under p = 2n; at n = 1600 even the non-private fit returns the zero
  vector. The design stays bounded and mean zero, which was its purpose.
* the zero estimator is reported, so a degenerate cell is visible.
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

from dp_primitives import analytic_gaussian_scale, wang_xu_label_iht_noise_scale

PROJECTION_RADIUS = np.float32(1.0)
ITERATIONS = 60
LAMBDA_MULTIPLIER = 0.75

# The three settings of the submitted paper, with the response bound fixed at the
# operating point already frozen for the label study. Setting 2 is rescaled to unit
# coordinate variance; see the module docstring.
SETTINGS = {
    "gaussian": dict(design="gaussian", noise="gaussian", noise_scale=1.0, coef="decay",
                     label_bound=1.615),
    "bounded_uniform": dict(design="bounded_uniform", noise="gaussian", noise_scale=1.0,
                            coef="decay", label_bound=1.615),
    "rademacher": dict(design="rademacher", noise="uniform", noise_scale=0.05,
                       coef="decay", label_bound=1.615),
}

SETTING_INDEX = {name: i for i, name in enumerate(SETTINGS)}

MODES = {
    "pilot": dict(seeds=[6101, 6102], sizes=[2000, 8000], sparsities=[5],
                  epsilons=[0.5, 0.95], bounds=[None], lambdas=[0.75]),
    # The response bound is the operating knob: a smaller bound trades clipping bias
    # against release noise, and the useful point lies inside the clipping region.
    # It is shared by both arms, so the release stays matched.
    "pilot_bound": dict(seeds=[6101, 6102], sizes=[2000, 8000], sparsities=[5],
                        epsilons=[0.95], bounds=[0.6, 0.9, 1.2, 1.615],
                        lambdas=[0.4, 0.75, 1.2]),
    # The coefficient profile decides whether the advantage persists as n grows: the
    # baseline is handed the true support size, so an equal-magnitude profile suits it,
    # while a decaying profile puts the smaller coordinates near the detection boundary
    # where shrinkage rather than a fixed count is the right response. Figure A1 of the
    # label study already records that the two are indistinguishable under a flat profile.
    "pilot_profile": dict(seeds=[6101, 6102], sizes=[8000, 16000], sparsities=[5],
                          epsilons=[0.95], bounds=[1.615], lambdas=[0.75],
                          profiles=["decay"]),
    "pilot_noise": dict(seeds=[6101, 6102], sizes=[2000, 8000], sparsities=[5],
                        epsilons=[0.95], bounds=[1.615], lambdas=[0.5, 0.75, 1.0],
                        noise_scales=[0.05, 0.25, 0.5]),
    # Seeds 6201-6208 were spent on the equal-magnitude profile, whose result is kept
    # as the recorded scope limit, so the decaying profile confirms on a fresh block.
    "confirm_flat": dict(seeds=[6201, 6202, 6203, 6204, 6205, 6206, 6207, 6208],
                         sizes=[2000, 4000, 8000, 16000], sparsities=[5],
                         epsilons=[0.95], bounds=[None], lambdas=[0.75],
                         profiles=["normal"]),
    "confirm": dict(seeds=[6301, 6302, 6303, 6304, 6305, 6306, 6307, 6308],
                    sizes=[2000, 4000, 8000, 16000], sparsities=[5], epsilons=[0.95],
                    bounds=[None], lambdas=[0.75]),
}


def soft(v, level):
    return np.sign(v) * np.maximum(np.abs(v) - level, np.float32(0.0))


def hard(v, keep):
    out = np.zeros_like(v)
    idx = np.argpartition(np.abs(v), -keep)[-keep:]
    out[idx] = v[idx]
    return out


def project(v):
    norm = np.linalg.norm(v)
    return v * (PROJECTION_RADIUS / norm) if norm > PROJECTION_RADIUS else v


def make_data(rng, n, p, s_star, spec):
    if spec["design"] == "gaussian":
        x = rng.standard_normal((n, p), dtype=np.float32)
    elif spec["design"] == "bounded_uniform":
        x = rng.random((n, p), dtype=np.float32)
        x *= np.float32(2 * math.sqrt(3.0))
        x -= np.float32(math.sqrt(3.0))
    else:
        x = rng.integers(0, 2, size=(n, p), dtype=np.int8).astype(np.float32)
        x *= np.float32(2.0)
        x -= np.float32(1.0)
    beta = np.zeros(p, dtype=np.float32)
    support = rng.choice(p, size=s_star, replace=False)
    if spec["coef"] == "normal":
        beta[support] = rng.standard_normal(s_star)
    elif spec["coef"] == "uniform":
        beta[support] = rng.uniform(0.0, 1.0, size=s_star)
    else:
        beta[support] = np.arange(1, s_star + 1, dtype=np.float32) ** -1.5
    beta /= np.linalg.norm(beta)
    noise = (rng.standard_normal(n) if spec["noise"] == "gaussian"
             else rng.uniform(-1.0, 1.0, size=n)) * spec["noise_scale"]
    return x, (x @ beta + noise.astype(np.float32)).astype(np.float32), beta


def smoothness(x, rng):
    n, p = x.shape
    v = rng.standard_normal(p).astype(np.float32)
    v /= np.linalg.norm(v)
    for _ in range(25):
        w = (x.T @ (x @ v)) / np.float32(n)
        v = w / max(float(np.linalg.norm(w)), 1e-12)
    return max(float(v @ ((x.T @ (x @ v)) / np.float32(n))), 1e-8)


def descend(x, y, *, estimator, regularization, sparsity, smooth):
    n = np.float32(x.shape[0])
    beta = np.zeros(x.shape[1], dtype=np.float32)
    step = np.float32(1.0 / smooth)
    level = np.float32(regularization / smooth)
    for _ in range(ITERATIONS):
        beta = beta - step * ((x.T @ (x @ beta - y)) / n)
        beta = soft(beta, level) if estimator == "soft" else hard(beta, sparsity)
        beta = project(beta)
    return beta


def run_cell(seed, n, s_star, epsilon, name, spec, bound, lambdas):
    p = 2 * n
    delta = 10.0 / n**1.1
    bound = spec["label_bound"] if bound is None else bound
    rng = np.random.default_rng([seed, n, s_star, int(epsilon * 1000),
                                 SETTING_INDEX[name], int(bound * 1000)])
    x, y, beta_star = make_data(rng, n, p, s_star, spec)
    smooth = smoothness(x, rng)

    sigma = analytic_gaussian_scale(2.0 * bound, epsilon, delta)
    clipped = np.clip(y, -bound, bound)
    released = (clipped + np.float32(sigma) * rng.standard_normal(n, dtype=np.float32))
    clip_rate = float(np.mean(np.abs(y) > bound))

    base_rate = math.sqrt(2.0 * math.log(p) / n)
    lam_plain = LAMBDA_MULTIPLIER * base_rate

    common = dict(seed=seed, n=n, p=p, s_star=s_star, epsilon=epsilon, delta=delta,
                  setting=name, noise_scale=spec["noise_scale"],
                  coef=spec["coef"],
                  smoothness=smooth, sigma=sigma, label_bound=bound,
                  clip_rate=clip_rate,
                  published_sigma=wang_xu_label_iht_noise_scale(bound, epsilon, delta))
    fits = [
        ("Non-private", math.nan, descend(x, y, estimator="soft",
                                          regularization=lam_plain, sparsity=0,
                                          smooth=smooth)),
        ("Wang-Xu Label-LDP-IHT", math.nan, descend(x, released, estimator="hard",
                                                    regularization=0.0,
                                                    sparsity=s_star, smooth=smooth)),
        ("Zero estimator", math.nan, np.zeros(p, dtype=np.float32)),
    ]
    for multiplier in lambdas:
        fits.append(("Label-LDP Lasso (ours)", multiplier,
                     descend(x, released, estimator="soft",
                             regularization=multiplier * sigma * base_rate,
                             sparsity=0, smooth=smooth)))
    return [{**common, "method": method, "multiplier": multiplier,
             "l2_error": float(np.linalg.norm(beta - beta_star)),
             "support": int(np.count_nonzero(beta))}
            for method, multiplier, beta in fits]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=sorted(MODES), default="pilot")
    args = parser.parse_args()
    spec = MODES[args.mode]
    started = time.perf_counter()

    rows = []
    for seed in spec["seeds"]:
        for n in spec["sizes"]:
            for s_star in spec["sparsities"]:
                for epsilon in spec["epsilons"]:
                    for name, setting in SETTINGS.items():
                        for bound in spec["bounds"]:
                            for scale in spec.get("noise_scales", [None]):
                                for profile in spec.get("profiles", [None]):
                                    variant = dict(setting)
                                    if scale is not None:
                                        variant["noise_scale"] = scale
                                    if profile is not None:
                                        variant["coef"] = profile
                                    rows += run_cell(seed, n, s_star, epsilon, name,
                                                     variant, bound, spec["lambdas"])
            print(f"  seed {seed} n={n} at {time.perf_counter() - started:.0f}s", flush=True)

    out = Path(__file__).resolve().parent / "results" / f"label_v1format_{args.mode}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    out.with_suffix(".json").write_text(json.dumps({
        "phase": "confirmation" if args.mode == "confirm" else "exploration",
        "mode": args.mode, **spec, "layout": "first-submission layout, p = 2n",
        "settings": SETTINGS, "iterations": ITERATIONS,
        "lambda_rule": f"{LAMBDA_MULTIPLIER} * sigma * sqrt(2 log p / n)",
        "delta_rule": "10/n^1.1", "calibration": "analytic Gaussian (Balle-Wang 2018)",
        "matched_release": "one released response vector per cell, shared by both arms",
    }, indent=2), encoding="utf-8")
    print(f"wrote {out} in {(time.perf_counter() - started)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
