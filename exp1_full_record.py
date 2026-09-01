"""Experiment 1: sequential full-record local privacy, ours against Zhu et al. (2024).

Both procedures are sequentially interactive, spend one message per user on disjoint
batches, and call the same bounded-vector randomizer at the same pure epsilon.  Within
a round they see the identical records.  They differ in exactly two places, which is
what the comparison isolates:

  radius   ours clips the whole score at G ~ sqrt(p log n); Zhu et al. truncate
           coordinatewise and use the deterministic bound r = sqrt(p) tau1
           (sqrt(k') tau1 + tau2), which carries an extra sqrt(k')
  update   ours takes a soft-threshold proximal step; Zhu et al. take a gradient step,
           keep the k' largest coordinates, and project onto the unit ball

Theory predicts errors of order sqrt(s* p)/(eps sqrt(n)) and s* sqrt(p)/(eps sqrt(n)),
so the ratio should widen like sqrt(s*).  Mode `sparsity` is the test of that.

The design is isotropic Gaussian, the one setting where both theorems apply:
Zhu et al. Theorem 7 assumes Sigma = I, and our Assumption A2 holds with mu = L = 1.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

import ldp_vector

DELTA_STAT = 0.05
NOISE_SD = 0.5
KAPPA_X = 1.0
KAPPA_Y = math.sqrt(1.0 + NOISE_SD**2)
RADIUS_PUBLIC = 2.0

MODES = {
    # Both radii are swept downward: a smaller radius trades clipping or truncation
    # bias against randomizer noise, and the useful operating point for either method
    # lies inside its own bias region.
    "feasible": dict(
        seeds=[5101], sizes=[200_000], dims=[20, 50, 100], sparsities=[5], epsilons=[1.0],
        c_clip=[0.02, 0.05, 0.1, 0.25], c_lam=[0.1, 0.25, 0.5, 1.0],
        c_trunc=[0.1, 0.25, 0.5, 1.0],
    ),
    # Does the frozen constant travel across n? If the functional form in the theory is
    # right, the same multiplier should be near-optimal at both sample sizes.
    "refine": dict(
        seeds=[5101], sizes=[200_000, 800_000], dims=[50], sparsities=[5], epsilons=[1.0],
        c_clip=[0.05, 0.075, 0.1, 0.15], c_lam=[0.05, 0.1, 0.15, 0.25],
        c_trunc=[0.05, 0.1, 0.15, 0.2],
    ),
    # The comparator's own tuning range. Its theory sets tau ~ sqrt(log n), so the
    # optimum must be allowed to move up with n; the grid is wide enough that its optimum stays interior.
    "zhu_range": dict(
        seeds=[5101], sizes=[800_000, 3_200_000], dims=[50], sparsities=[5], epsilons=[1.0],
        c_clip=[0.05], c_lam=[0.05], c_trunc=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
    ),
    # Frozen-constant check across the sparsity axis, with both grids narrowed around
    # their optima and the comparator's threshold level swept as well.
    "pilot_s": dict(
        seeds=[5101], sizes=[1_600_000], dims=[100], sparsities=[2, 4, 8], epsilons=[1.0],
        c_clip=[0.03, 0.05, 0.08], c_lam=[0.03, 0.05, 0.08],
        c_trunc=[0.05, 0.1, 0.25], keep_mult=[2, 4, 8],
    ),
    # Does a larger sample bring the comparator out of saturation at the top of the
    # sparsity range? Without that the s* exponent cannot be read off.
    "pilot_big": dict(
        seeds=[5101], sizes=[6_400_000], dims=[100], sparsities=[2, 4, 8], epsilons=[1.0],
        c_clip=[0.03, 0.05], c_lam=[0.03, 0.05],
        c_trunc=[0.05, 0.1], keep_mult=[2, 4],
    ),
    "pilot": dict(
        seeds=[5101, 5102], sizes=[200_000], dims=[200], sparsities=[5], epsilons=[1.0],
        c_clip=[0.25, 0.5, 1.0], c_lam=[0.5, 1.0, 2.0], c_trunc=[1.0],
    ),
    # Confirmation. Our constants are frozen from the pilots at p = 100, which selected
    # 0.03 for both at s* = 4 and 8 across n = 1.6e6 and 6.4e6. The comparator keeps its
    # threshold-level sweep and is credited with its best cell, as elsewhere in this
    # study. The axis is the sample size at fixed parameters: this is a head-to-head
    # between two procedures, not a measurement of either rate.
    "confirm": dict(
        seeds=[5301, 5302, 5303, 5304, 5305, 5306, 5307, 5308],
        sizes=[800_000, 1_600_000, 3_200_000, 6_400_000], dims=[100],
        sparsities=[4, 8], epsilons=[1.0],
        c_clip=[0.03], c_lam=[0.03], c_trunc=[0.1], keep_mult=[2, 4, 8],
    ),
}


def soft_threshold(v, level):
    return np.sign(v) * np.maximum(np.abs(v) - level, 0.0)


def hard_threshold(v, keep):
    out = np.zeros_like(v)
    idx = np.argpartition(np.abs(v), -keep)[-keep:]
    out[idx] = v[idx]
    return out


def project_ball(v, radius=1.0):
    norm = np.linalg.norm(v)
    return v * (radius / norm) if norm > radius else v


def draw_batch(rng, m, p, beta):
    x = rng.normal(size=(m, p))
    y = x @ beta + NOISE_SD * rng.normal(size=m)
    return x, y


def run_trial(seed, n, p, s_star, epsilon, clip_grid, lam_grid, trunc_grid, keep_grid):
    """One trial. Every configuration consumes the same batch stream, so the whole
    grid and both methods are compared on identical records."""
    rng = np.random.default_rng(seed)
    support = rng.choice(p, size=s_star, replace=False)
    beta_star = np.zeros(p)
    beta_star[support] = rng.normal(size=s_star)
    beta_star /= np.linalg.norm(beta_star)

    K = max(1, math.ceil(math.log(n)))            # L / mu = 1 for an isotropic design
    m = n // K
    step = 1.0                                    # 1 / L
    scale = KAPPA_X * (KAPPA_X * RADIUS_PUBLIC + KAPPA_Y)

    t_log = math.log(n / DELTA_STAT)
    ell = math.log(p * K / DELTA_STAT)
    d_m = (math.sqrt(ell / m) + ell / m
           + math.sqrt((p + t_log) * t_log * ell) / (epsilon * math.sqrt(m)))
    lam_plain = math.sqrt(2.0 * math.log(p) / m)

    ours = {(cc, cl): dict(beta=np.zeros(p), clip=cc * scale * math.sqrt((p + t_log) * t_log),
                           lam=cl * scale * d_m, clipped=0, peak=0.0)
            for cc in clip_grid for cl in lam_grid}
    # Their theory fixes k' = 8 s*, but the multiplier is swept so the comparator is
    # credited with its own best cell, matching how our constants are chosen.
    zhu = {}
    for ct in trunc_grid:
        for km in keep_grid:
            tau = ct * math.sqrt(math.log(n))
            keep = min(p, km * s_star)
            zhu[(ct, km)] = dict(beta=np.zeros(p), tau=tau, keep=keep,
                                 radius=math.sqrt(p) * tau * (math.sqrt(keep) + 1.0) * tau)
    plain = np.zeros(p)

    for _ in range(K):
        x, y = draw_batch(rng, m, p, beta_star)
        for state in ours.values():
            score = x * (x @ state["beta"] - y)[:, None]
            norms = np.linalg.norm(score, axis=1)
            state["clipped"] += int(np.count_nonzero(norms > state["clip"]))
            capped = score * np.minimum(1.0, state["clip"] / np.maximum(norms, 1e-12))[:, None]
            noisy = ldp_vector.randomize(capped, state["clip"], epsilon, rng).mean(axis=0)
            state["beta"] = soft_threshold(state["beta"] - step * noisy, step * state["lam"])
            state["peak"] = max(state["peak"], float(np.linalg.norm(state["beta"])))
        for state in zhu.values():
            xt = np.clip(x, -state["tau"], state["tau"])
            yt = np.clip(y, -state["tau"], state["tau"])
            grad = xt * (xt @ state["beta"] - yt)[:, None]
            noisy = ldp_vector.randomize(grad, state["radius"], epsilon, rng).mean(axis=0)
            state["beta"] = project_ball(hard_threshold(state["beta"] - step * noisy, state["keep"]))
        plain = soft_threshold(plain - step * (x * (x @ plain - y)[:, None]).mean(axis=0),
                               step * lam_plain)

    common = dict(seed=seed, n=n, p=p, s_star=s_star, epsilon=epsilon, K=K, m=m)
    rows = []
    for (cc, cl), state in ours.items():
        rows.append({**common, "method": "Full-Record LDP Lasso (ours)", "c_clip": cc,
                     "c_lam": cl, "c_trunc": math.nan, "keep": math.nan, "radius": state["clip"],
                     "lam": state["lam"], "clipped_scores": state["clipped"],
                     "peak_norm": state["peak"],
                     "l2_error": float(np.linalg.norm(state["beta"] - beta_star)),
                     "support": int(np.count_nonzero(state["beta"]))})
    for (ct, km), state in zhu.items():
        rows.append({**common, "method": "Zhu et al. LDP-IHT", "c_clip": math.nan,
                     "c_lam": km, "c_trunc": ct, "keep": state["keep"],
                     "radius": state["radius"],
                     "lam": math.nan, "clipped_scores": 0, "peak_norm": math.nan,
                     "l2_error": float(np.linalg.norm(state["beta"] - beta_star)),
                     "support": int(np.count_nonzero(state["beta"]))})
    for name, value in (("Non-private proximal", plain), ("Zero estimator", np.zeros(p))):
        rows.append({**common, "method": name, "c_clip": math.nan, "c_lam": math.nan,
                     "c_trunc": math.nan, "keep": math.nan, "radius": math.nan, "lam": math.nan,
                     "clipped_scores": 0, "peak_norm": math.nan,
                     "l2_error": float(np.linalg.norm(value - beta_star)),
                     "support": int(np.count_nonzero(value))})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=sorted(MODES), default="pilot")
    args = parser.parse_args()
    spec = MODES[args.mode]
    started = time.perf_counter()

    rows = []
    for seed in spec["seeds"]:
        for n in spec["sizes"]:
            for p_dim in spec["dims"]:
                for s_star in spec["sparsities"]:
                    for epsilon in spec["epsilons"]:
                        rows += run_trial(seed, n, p_dim, s_star, epsilon,
                                          spec["c_clip"], spec["c_lam"], spec["c_trunc"],
                                          spec.get("keep_mult", [8]))
                        print(f"  seed {seed} p={p_dim} s*={s_star} eps={epsilon} "
                              f"at {time.perf_counter() - started:.0f}s", flush=True)

    out = Path(__file__).resolve().parent / "results" / f"exp1_{args.mode}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    out.with_suffix(".json").write_text(json.dumps({
        "phase": "confirmation" if args.mode == "sparsity" else "exploration",
        "mode": args.mode, **spec, "delta_stat": DELTA_STAT, "noise_sd": NOISE_SD,
        "design": "isotropic Gaussian; Sigma = I so mu = L = 1",
        "randomizer": "Duchi et al. l2-ball mechanism, shared by both procedures",
        "matching": "identical records per round; K = ceil(log n) batches of size n//K",
        "zhu_note": "Trunc applied to the freshly formed iterate, per the text; arXiv v1 "
                    "Algorithm 2 line 6 prints a subscript t-1",
    }, indent=2), encoding="utf-8")
    print(f"wrote {out} in {(time.perf_counter() - started)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
