"""
model/als.py
============
Alternating Least Squares (ALS) bilinear factorisation for GPT rank estimation.

Model
-----
We seek a rank-K factorisation of the probability matrix D (shape S x X):

    D ≈ U @ V.T,   U in R^{S x K},  V in R^{X x K}

Loss Function (Poisson-weighted chi-squared)
--------------------------------------------
Minimise:

    L(U, V) = sum_{s,x} w[s,x] * (D[s,x] - (U @ V.T)[s,x])^2

where:
    w[s,x]    = 1 / sigma[s,x]^2
    sigma[s,x] = max( sqrt(|D[s,x]| / N_eff),  SIGMA_FLOOR )

ALS Update Rules (Normal Equations)
-------------------------------------
Update U row-by-row (fix V):

    A[s, k, l] = sum_x w[s,x] * V[x,k] * V[x,l]  +  reg * delta_{kl}
    b[s, k]    = sum_x w[s,x] * D[s,x] * V[x,k]
    u[s, :]    = solve(A[s], b[s])

Update V row-by-row (fix U):

    A[x, k, l] = sum_s w[s,x] * U[s,k] * U[s,l]  +  reg * delta_{kl}
    b[x, k]    = sum_s w[s,x] * D[s,x] * U[s,k]
    v[x, :]    = solve(A[x], b[x])

SVD Warm-Start
--------------
Restart 0 uses the leading K left/right singular vectors of D as initialisation.
Restarts 1..N_RESTARTS-1 add Gaussian perturbations scaled by 0.1 * U_init.std().

Cross-Validation
----------------
als_fit accepts a boolean train_mask (True = training pixel). The test chi-squared
is computed on the held-out (False) entries after fitting.
"""

import numpy as np
from numpy.linalg import svd, solve

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42
ALS_REG: float = 1e-6       # L2 regularisation on U and V
MAX_ITER: int = 500         # Maximum ALS iterations per restart
TOL: float = 1e-7           # Relative convergence tolerance
N_RESTARTS: int = 4         # Number of random restarts
SIGMA_FLOOR: float = 1e-4   # Minimum sigma (prevents infinite weights)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def poisson_sigma(prob_mat: np.ndarray, n_eff: float) -> np.ndarray:
    """Compute Poisson standard deviation for each (setting, pixel) entry.

    Parameters
    ----------
    prob_mat : np.ndarray, shape (S, X)
        Probability matrix D[s, x] (rows sum to 1).
    n_eff : float
        Effective photon count per measurement setting.

    Returns
    -------
    sigma : np.ndarray, shape (S, X)
        sigma[s, x] = max( sqrt(|p[s,x]| / n_eff),  SIGMA_FLOOR ).
    """
    sigma = np.sqrt(np.abs(prob_mat) / float(n_eff))
    return np.maximum(sigma, SIGMA_FLOOR)


# ---------------------------------------------------------------------------
# Core ALS fitting
# ---------------------------------------------------------------------------

def als_fit(
    data: np.ndarray,
    sigma: np.ndarray,
    train_mask: np.ndarray,
    K: int,
    reg: float = ALS_REG,
    n_restarts: int = N_RESTARTS,
    seed: int = RANDOM_SEED,
) -> tuple:
    """Fit a rank-K bilinear model  D ≈ U @ V.T  via Alternating Least Squares.

    Parameters
    ----------
    data : np.ndarray, shape (S, X)
        Observed probability matrix (may be noisy).
    sigma : np.ndarray, shape (S, X)
        Per-entry standard deviation (from poisson_sigma).
    train_mask : np.ndarray of bool, shape (S, X)
        True for training entries, False for held-out test entries.
    K : int
        Rank of the factorisation.
    reg : float
        L2 regularisation coefficient. Default: ALS_REG.
    n_restarts : int
        Number of random restarts. The best (lowest training chi-squared)
        result is returned. Default: N_RESTARTS.
    seed : int
        Base random seed; each restart uses seed + restart_index.

    Returns
    -------
    U : np.ndarray, shape (S, K)
        Left factor (state matrix).
    V : np.ndarray, shape (X, K)
        Right factor (effect/pixel matrix); shared across groups in joint fit.
    train_chi2 : float
        Training chi-squared per degree of freedom.
    test_chi2 : float
        Test chi-squared per degree of freedom.

    Raises
    ------
    ValueError
        If K > min(S, X).
    """
    S, X = data.shape
    if K > min(S, X):
        raise ValueError(f'K={K} exceeds min(S={S}, X={X})')

    # Precompute weights (zero out test entries during training)
    w_full = 1.0 / sigma ** 2  # shape (S, X)
    w_train = w_full * train_mask.astype(float)
    w_test = w_full * (~train_mask).astype(float)

    # SVD warm-start initialisation
    # Use train-masked data (fill test entries with row mean to avoid bias)
    data_filled = data.copy()
    row_means = (data * train_mask).sum(axis=1, keepdims=True) / (
        train_mask.sum(axis=1, keepdims=True).clip(min=1)
    )
    data_filled[~train_mask] = row_means[~train_mask]

    Us, ss, Vts = svd(data_filled, full_matrices=False)
    # Clamp K to available singular values
    K_eff = min(K, len(ss))
    U_svd = Us[:, :K_eff] * ss[:K_eff]   # absorb singular values into U
    V_svd = Vts[:K_eff].T                 # shape (X, K_eff)
    # Pad with zeros if K_eff < K
    if K_eff < K:
        U_svd = np.hstack([U_svd, np.zeros((S, K - K_eff))])
        V_svd = np.hstack([V_svd, np.zeros((X, K - K_eff))])

    I_K = np.eye(K)
    best_U, best_V = None, None
    best_train_chi2 = np.inf

    for restart in range(n_restarts):
        rng_r = np.random.default_rng(seed + restart)

        # Initialise U and V
        if restart == 0:
            U = U_svd.copy()
            V = V_svd.copy()
        else:
            noise_scale = 0.1 * (U_svd.std() + 1e-12)
            U = U_svd + rng_r.standard_normal(U_svd.shape) * noise_scale
            V = V_svd + rng_r.standard_normal(V_svd.shape) * noise_scale

        prev_chi2 = np.inf

        for iteration in range(MAX_ITER):
            # ----------------------------------------------------------
            # Update U: solve for each row s
            # A[s, k, l] = sum_x w_train[s,x] * V[x,k] * V[x,l] + reg * I
            # b[s, k]    = sum_x w_train[s,x] * data[s,x] * V[x,k]
            # ----------------------------------------------------------
            # A_skl shape: (S, K, K)
            A_U = np.einsum('sx,xk,xl->skl', w_train, V, V) + reg * I_K[None, :, :]
            b_U = np.einsum('sx,sx,xk->sk', w_train, data, V)
            # Solve S independent K x K systems
            for s in range(S):
                try:
                    U[s] = solve(A_U[s], b_U[s])
                except np.linalg.LinAlgError:
                    # Singular matrix: keep current U[s]
                    pass

            # ----------------------------------------------------------
            # Update V: solve for each pixel x
            # A[x, k, l] = sum_s w_train[s,x] * U[s,k] * U[s,l] + reg * I
            # b[x, k]    = sum_s w_train[s,x] * data[s,x] * U[s,k]
            # ----------------------------------------------------------
            A_V = np.einsum('sx,sk,sl->xkl', w_train, U, U) + reg * I_K[None, :, :]
            b_V = np.einsum('sx,sx,sk->xk', w_train, data, U)
            for x in range(X):
                try:
                    V[x] = solve(A_V[x], b_V[x])
                except np.linalg.LinAlgError:
                    pass

            # ----------------------------------------------------------
            # Compute training chi-squared / dof
            # ----------------------------------------------------------
            D_hat = U @ V.T
            residuals = (data - D_hat) * np.sqrt(w_train)
            n_train = train_mask.sum()
            curr_chi2 = (residuals ** 2).sum() / max(n_train - S * K - X * K, 1)

            # Convergence check
            delta = abs(prev_chi2 - curr_chi2) / max(abs(prev_chi2), 1e-12)
            if delta < TOL:
                break
            prev_chi2 = curr_chi2

        # Keep best restart
        if curr_chi2 < best_train_chi2:
            best_train_chi2 = curr_chi2
            best_U = U.copy()
            best_V = V.copy()

    # ------------------------------------------------------------------
    # Compute test chi-squared using best U, V
    # ------------------------------------------------------------------
    D_hat_best = best_U @ best_V.T
    n_test = (~train_mask).sum()
    if n_test > 0:
        res_test = (data - D_hat_best) * np.sqrt(w_test)
        test_chi2 = (res_test ** 2).sum() / max(n_test, 1)
    else:
        test_chi2 = 0.0

    return best_U, best_V, best_train_chi2, test_chi2
