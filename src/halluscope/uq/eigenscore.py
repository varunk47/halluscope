"""EigenScore, implemented from Chen et al., INSIDE: LLMs' Internal States
Retain the Power of Hallucination Detection (ICLR 2024, arXiv:2402.03744).

Given K sentence embeddings z_1..z_K in R^d stacked as Z (d x K), the paper
forms the covariance Sigma = Z^T J_d Z with the centering matrix
J_d = I_d - (1/d) 1 1^T, regularizes with alpha I_K, and reports
(1/K) sum_i log lambda_i. Larger values mean the K responses spread across
more semantic directions, which the paper reads as hallucination.

Centering with J_d is equivalent to subtracting each embedding's own mean
across features, so we do that directly instead of materializing a d x d
matrix.
"""

from __future__ import annotations

import numpy as np


def eigenscore(embeddings: np.ndarray, alpha: float = 1e-3) -> float:
    """``embeddings`` is (K, d). Returns the mean log eigenvalue of the K x K covariance."""
    Z = np.asarray(embeddings, dtype=np.float64)
    if Z.ndim != 2 or Z.shape[0] < 2:
        raise ValueError("need at least two embeddings of shape (K, d)")
    K = Z.shape[0]
    Zc = Z - Z.mean(axis=1, keepdims=True)  # J_d applied per embedding
    sigma = Zc @ Zc.T  # (K, K)
    sigma = sigma + alpha * np.eye(K)
    eig = np.linalg.eigvalsh(sigma)
    eig = np.clip(eig, a_min=1e-12, a_max=None)
    return float(np.mean(np.log(eig)))


def eigen_spectrum(embeddings: np.ndarray, alpha: float = 1e-3) -> np.ndarray:
    Z = np.asarray(embeddings, dtype=np.float64)
    Zc = Z - Z.mean(axis=1, keepdims=True)
    sigma = Zc @ Zc.T + alpha * np.eye(Z.shape[0])
    return np.sort(np.linalg.eigvalsh(sigma))[::-1]
