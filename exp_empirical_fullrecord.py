"""Full-record local privacy on real records: our procedure against Zhu et al. (2024).

The full-record rate carries `sqrt(p)`, so its usable regime on real data is a moderate
number of features with many records, not a design with tens of thousands of columns.
The ACS income task is therefore built here from the low-cardinality covariates only,
giving `p` in the low hundreds against `n` in the hundreds of thousands. The
high-dimensional version of the same task is the label-private case.

Rows are normalized to unit `l2` norm and responses are +-1, so on this design our
whole-score radius is exact and public without any dimension factor:

    ||x_i (x_i' beta - y_i)||_2 = |x_i' beta - y_i| <= ||beta|| ||x_i|| + 1 <= 2.

Zhu et al. truncate coordinatewise and use the deterministic bound
`sqrt(p) tau (sqrt(k') tau + tau)`, which is 27 to 52 times larger here. Because that is
a property of their prescribed radius rather than of their update rule, a third arm runs
their update on our radius, which isolates hard against soft thresholding on identical
messages. All arms share the batches and the randomizer at the same pure epsilon.
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
from sklearn.preprocessing import OneHotEncoder, normalize

import ldp_vector

FEATURES = ["AGEP", "COW", "SCHL", "MAR", "SEX", "RAC1P", "WKHP", "OCCP"]
READ = FEATURES + ["PINCP", "PWGTP"]
BINNED = {"AGEP": 5, "WKHP": 5, "OCCP": 500}     # coarse bands keep the design moderate
TEST_SIZE = 20_000
PROJECTION_RADIUS = 1.0
SCORE_RADIUS = 2.0                                # exact on a unit-norm row design

MODES = {
    "pilot": dict(seeds=[7601, 7602], sizes=[150_000, 400_000], epsilons=[1.0],
                  lambdas=[0.01, 0.02, 0.035, 0.05, 0.08],
                  taus=[0.03, 0.05, 0.07, 0.1, 0.15, 0.25],
                  keeps=[10, 25, 50, 100]),
    # The national pool allows a sample four times larger. The comparator's truncation
    # grid widens with it: its own theory sets tau ~ sqrt(log n), so its optimum drifts
    # up, and freezing the range chosen at 400,000 records would pin it in its bias
    # branch. The grid is otherwise narrow because each arm touches m by p entries per
    # round and there are fourteen rounds at the largest size.
    "pilot_national": dict(seeds=[7801, 7802], sizes=[400_000, 1_600_000],
                           epsilons=[1.0], lambdas=[0.02, 0.05],
                           taus=[0.03, 0.07, 0.15, 0.3, 0.6], keeps=[25, 100]),
    # The multiplier reached the top of the previous grid, so it is located before the
    # confirmation freezes it. The comparator is reduced to one cell here because only
    # our own tuning is in question.
    "pilot_lambda": dict(seeds=[7801, 7802], sizes=[1_600_000], epsilons=[1.0],
                         lambdas=[0.035, 0.05, 0.08, 0.12, 0.18],
                         taus=[0.07], keeps=[100]),
    "confirm": dict(seeds=[7701, 7702, 7703, 7704, 7705, 7706, 7707, 7708],
                    sizes=[200_000, 400_000, 800_000, 1_600_000], epsilons=[1.0],
                    lambdas=[None], taus=[0.03, 0.07, 0.15, 0.3, 0.6],
                    keeps=[10, 25, 50, 100]),
}


def soft(v, level):
    return np.sign(v) * np.maximum(np.abs(v) - level, 0.0)


def hard(v, keep):
    out = np.zeros_like(v)
    idx = np.argpartition(np.abs(v), -keep)[-keep:]
    out[idx] = v[idx]
    return out


def project(v):
    norm = np.linalg.norm(v)
    return v * (PROJECTION_RADIUS / norm) if norm > PROJECTION_RADIUS else v


EXTRACT = Path(__file__).resolve().parent / "data" / "acs2018_income.parquet"


def load_public():
    """The pooled ACS income extract built by `fetch_acs_national.py`."""
    if not EXTRACT.exists():
        raise SystemExit(
            f"{EXTRACT} not found. Run `python fetch_acs_national.py` first; it "
            "downloads the 2018 ACS 1-Year person files and writes the extract.")
    return pd.read_parquet(EXTRACT)[READ]


def design(values):
    base = values.astype(np.int64, copy=True)
    for name, width in BINNED.items():
        base[:, FEATURES.index(name)] //= width
    return base


def one_hot(train_values, test_values):
    combined = np.vstack([design(train_values), design(test_values)])
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=float)
    matrix = normalize(encoder.fit_transform(combined).tocsr(), norm="l2", axis=1,
                       copy=False)
    return matrix[:len(train_values)], matrix[len(train_values):]


def smoothness(matrix, seed):
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=matrix.shape[1])
    vector /= np.linalg.norm(vector)
    n = matrix.shape[0]
    for _ in range(35):
        moved = np.asarray(matrix.T @ (matrix @ vector)).ravel() / n
        vector = moved / max(float(np.linalg.norm(moved)), 1e-12)
    return max(float(vector @ (np.asarray(matrix.T @ (matrix @ vector)).ravel() / n)) * 1.05,
               1e-8)


def nmse(matrix, response, beta, train_mean):
    prediction = np.asarray(matrix @ beta).ravel()
    return float(np.mean((prediction - response) ** 2)
                 / np.mean((response - train_mean) ** 2))


def scores(batch_x, batch_y, beta):
    return batch_x.multiply((batch_x @ beta - batch_y)[:, None]).toarray()


def run_seed(seed, spec, public, labels):
    largest = max(spec["sizes"])
    rng = np.random.default_rng(seed + 700_000)
    index = rng.choice(len(public), size=largest + TEST_SIZE, replace=False)
    features = public.iloc[index][FEATURES].to_numpy()
    train_x, test_x = one_hot(features[:largest], features[largest:])
    train_y, test_y = labels[index][:largest], labels[index][largest:]
    p = train_x.shape[1]
    rows = []

    for n in spec["sizes"]:
        x_all, y_all = train_x[:n], train_y[:n]
        mean_n = float(y_all.mean())
        smooth = smoothness(x_all, seed=seed + 900_000 + n)
        step = 1.0 / smooth
        K = max(1, math.ceil(math.log(n)))
        m = n // K
        ell = math.log(p * K / 0.05)
        base = math.sqrt(ell / m) + SCORE_RADIUS * math.sqrt(p * ell / m)
        common = dict(seed=seed, n=n, p=p, K=K, m=m, smoothness=smooth,
                      test_size=len(test_y))

        plain = np.zeros(p)
        plain_level = math.sqrt(2.0 * math.log(p) / n)
        for _ in range(150):
            grad = np.asarray(x_all.T @ (x_all @ plain - y_all)).ravel() / n
            plain = project(soft(plain - step * grad, step * plain_level))
        rows.append({**common, "epsilon": math.nan, "method": "Non-private proximal",
                     "multiplier": math.nan, "tau": math.nan, "keep": math.nan,
                     "radius": math.nan, "nmse": nmse(test_x, test_y, plain, mean_n),
                     "support": int(np.count_nonzero(plain))})
        rows.append({**common, "epsilon": math.nan, "method": "Constant predictor",
                     "multiplier": math.nan, "tau": math.nan, "keep": math.nan,
                     "radius": math.nan,
                     "nmse": nmse(test_x, test_y, np.zeros(p), mean_n), "support": 0})

        for epsilon in spec["epsilons"]:
            arms = {}
            for multiplier in spec["lambdas"]:
                arms[("Full-Record LDP Lasso (ours)", multiplier, math.nan, math.nan)] = \
                    dict(beta=np.zeros(p), radius=SCORE_RADIUS,
                         level=multiplier * base, kind="soft", keep=0)
            # Their truncation acts on the record, once, before any round.
            truncated = {}
            for tau in spec["taus"]:
                xt = x_all.copy()
                np.clip(xt.data, -tau, tau, out=xt.data)
                truncated[tau] = (xt, y_all)          # tau2 = 1 is exact on +-1
            for keep in spec["keeps"]:
                arms[("Zhu et al. LDP-IHT, matched radius", math.nan, math.nan, keep)] = \
                    dict(beta=np.zeros(p), radius=SCORE_RADIUS, level=0.0,
                         kind="hard", keep=keep)
                for tau in spec["taus"]:
                    radius = math.sqrt(p) * tau * (math.sqrt(keep) * tau + 1.0)
                    arms[("Zhu et al. LDP-IHT", math.nan, tau, keep)] = \
                        dict(beta=np.zeros(p), radius=radius, level=0.0,
                             kind="hard", keep=keep, tau=tau)
            for index, state in enumerate(arms.values()):
                state["rng"] = np.random.default_rng([seed, n, index])
            for k in range(K):
                part = slice(k * m, (k + 1) * m)
                bx, by = x_all[part], y_all[part]
                for state in arms.values():
                    if "tau" in state:
                        # Their record-level truncation; the deterministic radius then
                        # bounds the gradient and nothing is clipped afterwards.
                        tx, ty = truncated[state["tau"]]
                        raw = scores(tx[part], ty[part], state["beta"])
                    else:
                        raw = scores(bx, by, state["beta"])
                        norms = np.linalg.norm(raw, axis=1)
                        raw = raw * np.minimum(1.0, state["radius"]
                                               / np.maximum(norms, 1e-12))[:, None]
                    noisy = ldp_vector.randomize(raw, state["radius"], epsilon,
                                                 state["rng"]).mean(axis=0)
                    moved = state["beta"] - step * noisy
                    state["beta"] = project(soft(moved, step * state["level"])
                                            if state["kind"] == "soft"
                                            else hard(moved, state["keep"]))
            for (method, multiplier, tau, keep), state in arms.items():
                rows.append({**common, "epsilon": epsilon, "method": method,
                             "multiplier": multiplier, "tau": tau, "keep": keep,
                             "radius": state["radius"],
                             "nmse": nmse(test_x, test_y, state["beta"], mean_n),
                             "support": int(np.count_nonzero(state["beta"]))})
        print(f"  seed={seed} n={n} p={p} K={K} m={m} at {time.perf_counter():.0f}", flush=True)
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
    print(f"loaded {len(public):,} rows in {time.perf_counter()-started:.0f}s; "
          f"positive rate {np.mean(labels > 0):.3f}", flush=True)

    rows = []
    for seed in spec["seeds"]:
        rows += run_seed(seed, spec, public, labels)

    out = (Path(__file__).resolve().parent / "results"
           / f"fullrecord_real_{args.mode}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    out.with_suffix(".json").write_text(json.dumps({
        "phase": "confirmation" if args.mode == "confirm" else "exploration",
        "mode": args.mode, **spec, "features": FEATURES, "binned": BINNED,
        "task": "ACS income above 50,000, CA/NY/TX 2018 1-Year",
        "privacy": "pure epsilon-LDP, sequentially interactive, disjoint batches",
        "score_radius": SCORE_RADIUS,
        "score_radius_note": "exact on a unit-norm row design with responses in {-1,+1}",
        "randomizer": "l2-ball mechanism, shared by every private arm",
        "metric": "test NMSE against the training-mean predictor",
    }, indent=2), encoding="utf-8")
    print(f"wrote {out} in {(time.perf_counter()-started)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
