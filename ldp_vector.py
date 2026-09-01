"""The bounded-vector pure-LDP randomizer shared by both sequential procedures.

This is the l2-ball mechanism of Duchi, Jordan and Wainwright (2018), which appears
as Definition 2 (`def:bounded-vector-randomizer`) in our Section 3.2 and as equation
(6) of Zhu et al. (2024).  Both procedures in Experiment 1 call it, so the comparison
differs only in the clipping radius each method prescribes and in the server update.

On input `v` with ||v||_2 <= r it returns an unbiased, pure eps-LDP message whose
coordinates are sub-Gaussian with norm O(r/eps).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import gammaln


def half_sphere_mean(p: int) -> float:
    """E<u,v> for u uniform on the half sphere {u in S^{p-1} : <u,v> > 0}."""
    return 2.0 * math.exp(gammaln(p / 2.0) - gammaln((p - 1) / 2.0)) / ((p - 1) * math.sqrt(math.pi))


def output_norm(radius: float, epsilon: float, p: int) -> float:
    """The scaling that makes the mechanism unbiased; also the message's l2 norm."""
    tilt = (math.exp(epsilon) - 1.0) / (math.exp(epsilon) + 1.0)
    return radius / (half_sphere_mean(p) * tilt)


def randomize(vectors: np.ndarray, radius: float, epsilon: float, rng) -> np.ndarray:
    """Apply the mechanism row-wise to an (m, p) array of vectors of norm at most `radius`."""
    m, p = vectors.shape
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Randomized rounding to the sphere of radius `radius`, which is unbiased.
    sign = np.where(rng.random((m, 1)) < 0.5 + norms / (2.0 * radius), 1.0, -1.0)
    anchor = sign * np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)

    direction = rng.normal(size=(m, p))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    # Uniform on the anchor's half sphere with probability e^eps/(e^eps+1), else the other.
    want = np.where(rng.random((m, 1)) < math.exp(epsilon) / (math.exp(epsilon) + 1.0), 1.0, -1.0)
    agrees = np.sign(np.sum(direction * anchor, axis=1, keepdims=True))
    direction *= np.where(agrees == 0.0, 1.0, agrees) * want
    return output_norm(radius, epsilon, p) * direction
