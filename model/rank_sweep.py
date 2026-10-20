"""
model/rank_sweep.py
===================
Rank sweep with cross-validation for GPT rank estimation.

Joint Fit Strategy
------------------
All n_open groups (n=1,2,3,4) are stacked into a single matrix:

    all_mat = vstack([mats[1], mats[2], mats[3], mats[4]])

A JOINT bilinear model  D ≈ U @ V.T  is fitted where V (the pixel/effect matrix)
is SHARED across all groups. The rows of U are group-specific but V encodes
the common detector response.

This enforces physical consistency: the detector pixel effects cannot depend
on which slits are open.

Cross-Validation
----------------
Folds are assigned randomly to (row, pixel) pairs to avoid block structure
bias. For each fold f:
    - Train on: fold_ids != f
    - Test on:  fold_ids == f
    - Compute test chi^2/dof for the entire stacked matrix
    - Also slice back per-group results

Return Format
-------------
results[group][K] = {
    'train': [fold0_chi2, fold1_chi2, ...],
    'test':  [fold0_chi2, fold1_chi2, ...],
}
where group in {1, 2, 3, 4, 'all'}.
"""

import numpy as np
from joblib import Parallel, delayed

from .als import als_fit, poisson_sigma

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_PX_SWEEP: int = 500   # Number of pixels to use in the sweep (subsample)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resample_cols(mat: np.ndarray, n_cols: int, seed: int) -> np.ndarray:
    """Randomly subsample columns of a matrix.

    Parameters
    ----------
    mat : np.ndarray, shape (S, X)
    n_cols : int
        Target number of columns. If n_cols >= X, returns mat unchanged.
    seed : int

    Returns
    -------
    np.ndarray, shape (S, min(n_cols, X))
    """
    S, X = mat.shape
    if n_cols >= X:
        return mat
    rng = np.random.default_rng(seed)
    col_idx = rng.choice(X, size=n_cols, replace=False)
    col_idx.sort()
    return mat[:, col_idx]


def _fit_one_fold(
    all_mat: np.ndarray,
    sigma_mat: np.ndarray,
    fold_ids: np.ndarray,
    K: int,
    fold: int,
    reg: float,
    group_row_slices: dict,
    seed: int,
) -> tuple:
    """Fit rank-K model for one CV fold and return per-group chi-squared.

    Parameters
    ----------
    all_mat : np.ndarray, shape (total_rows, n_pix)
        Stacked probability matrix for all n_open groups.
    sigma_mat : np.ndarray, shape (total_rows, n_pix)
        Per-entry Poisson sigma (same shape as all_mat).
    fold_ids : np.ndarray of int, shape (total_rows, n_pix)
        Fold assignment for each (row, pixel) entry.
    K : int
        Rank to fit.
    fold : int
        Current fold index.
    reg : float
        Regularisation coefficient.
    group_row_slices : dict
        {n_open: slice(row_start, row_end)} for each n_open group.
    seed : int
        Random seed for ALS restarts.

    Returns
    -------
    K : int
    fold : int
    group_chi2 : dict
        {n_open: (train_chi2, test_chi2), ..., 'all': (train_chi2, test_chi2)}
    """
    train_mask = (fold_ids != fold)  # bool, shape (total_rows, n_pix)

    try:
        U, V, train_chi2_all, test_chi2_all = als_fit(
            data=all_mat,
            sigma=sigma_mat,
            train_mask=train_mask,
            K=K,
            reg=reg,
            seed=seed,
        )
    except Exception:
        # Failed fit: return large chi2 sentinel
        group_chi2 = {'all': (1e6, 1e6)}
        for n_open, slc in group_row_slices.items():
            group_chi2[n_open] = (1e6, 1e6)
        return K, fold, group_chi2

    # Compute per-group chi-squared by slicing rows of U
    D_hat = U @ V.T  # shape (total_rows, n_pix)
    w_full = 1.0 / sigma_mat ** 2

    group_chi2 = {'all': (train_chi2_all, test_chi2_all)}

    for n_open, slc in group_row_slices.items():
        U_g = U[slc]          # shape (n_rows_g, K)
        D_hat_g = U_g @ V.T   # shape (n_rows_g, n_pix)
        data_g = all_mat[slc]
        sigma_g = sigma_mat[slc]
        w_g = w_full[slc]
        fold_g = fold_ids[slc]
        train_g = (fold_g != fold)
        test_g = (fold_g == fold)

        res = (data_g - D_hat_g) ** 2 * w_g

        n_tr = train_g.sum()
        n_ts = test_g.sum()
        tr_chi2 = (res * train_g).sum() / max(n_tr, 1)
        ts_chi2 = (res * test_g).sum() / max(n_ts, 1)

        group_chi2[n_open] = (tr_chi2, ts_chi2)

    return K, fold, group_chi2


# ---------------------------------------------------------------------------
# Public rank sweep
# ---------------------------------------------------------------------------

def run_rank_sweep(
    data: dict,
    K_range=range(1, 21),
    n_folds: int = 10,
    n_jobs: int = -1,
    seed: int = 42,
) -> dict:
    """Sweep over ranks K using cross-validated chi-squared to find GPT rank.

    Parameters
    ----------
    data : dict
        Output of InterferometerSimulator.simulate(): {n_open: {'matrix': ..., 'n_eff': ...}}.
    K_range : iterable of int
        Ranks to sweep. Default: range(1, 21).
    n_folds : int
        Number of cross-validation folds. Default: 10.
    n_jobs : int
        Number of parallel workers. -1 = all CPU cores. Default: -1.
    seed : int
        Random seed. Default: 42.

    Returns
    -------
    results : dict
        Nested dict: results[group][K] = {'train': [fold_values], 'test': [fold_values]}
        where group in {1, 2, 3, 4, 'all'}.
    """
    rng = np.random.default_rng(seed)
    K_list = list(K_range)
    n_open_list = sorted(data.keys())

    # ------------------------------------------------------------------
    # Subsample pixels
    # ------------------------------------------------------------------
    mats_r = {}
    n_effs = {}
    for n_open in n_open_list:
        mat = data[n_open]['matrix']
        mat_r = _resample_cols(mat, N_PX_SWEEP, seed)
        mats_r[n_open] = mat_r
        n_effs[n_open] = data[n_open]['n_eff']

    n_pix = mats_r[n_open_list[0]].shape[1]

    # ------------------------------------------------------------------
    # Build stacked matrix and group_row_slices
    # ------------------------------------------------------------------
    all_mats = []
    group_row_slices = {}
    row_ptr = 0
    for n_open in n_open_list:
        mat = mats_r[n_open]
        n_rows = mat.shape[0]
        group_row_slices[n_open] = slice(row_ptr, row_ptr + n_rows)
        all_mats.append(mat)
        row_ptr += n_rows

    all_mat = np.vstack(all_mats)   # shape (total_rows, n_pix)
    total_rows = all_mat.shape[0]

    # ------------------------------------------------------------------
    # Build sigma matrix (use per-group n_eff)
    # ------------------------------------------------------------------
    sigma_parts = []
    for n_open in n_open_list:
        mat_g = mats_r[n_open]
        n_eff_g = n_effs[n_open]
        sigma_parts.append(poisson_sigma(mat_g, n_eff_g))
    sigma_mat = np.vstack(sigma_parts)  # shape (total_rows, n_pix)

    # ------------------------------------------------------------------
    # Regularisation: scale with mean n_eff
    # ------------------------------------------------------------------
    mean_n_eff = np.mean(list(n_effs.values()))
    reg = max(ALS_REG_DEFAULT, mean_n_eff * 1e-8)

    # ------------------------------------------------------------------
    # Random fold assignment on (row, pixel) pairs
    # ------------------------------------------------------------------
    n_entries = total_rows * n_pix
    flat_idx = rng.permutation(n_entries)
    fold_ids = (flat_idx % n_folds).reshape(total_rows, n_pix).astype(np.int32)

    # ------------------------------------------------------------------
    # Print header
    # ------------------------------------------------------------------
    print(f'Rank sweep | matrix: {total_rows}x{n_pix} | '
          f'N_eff: {mean_n_eff:.0f} | reg: {reg:.2e} | n_jobs: {n_jobs}')
    print(f'K_range: {K_list[0]}..{K_list[-1]} | n_folds: {n_folds}')
    print(f'{"K":>4}  {"n=1 test χ²":>14}  {"n=2 test χ²":>14}  '
          f'{"all test χ²":>14}')
    print('-' * 55)

    # ------------------------------------------------------------------
    # Parallel CV sweep
    # ------------------------------------------------------------------
    tasks = [
        (K, f)
        for K in K_list
        for f in range(n_folds)
    ]

    raw_results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_fit_one_fold)(
            all_mat, sigma_mat, fold_ids, K, fold, reg, group_row_slices, seed + K * 1000 + fold
        )
        for K, fold in tasks
    )

    # ------------------------------------------------------------------
    # Collect results
    # ------------------------------------------------------------------
    all_groups = n_open_list + ['all']
    results = {}
    for grp in all_groups:
        results[grp] = {K: {'train': [], 'test': []} for K in K_list}

    for (K, fold, group_chi2) in raw_results:
        for grp in all_groups:
            tr, ts = group_chi2.get(grp, (1e6, 1e6))
            results[grp][K]['train'].append(tr)
            results[grp][K]['test'].append(ts)

    # ------------------------------------------------------------------
    # Print per-K summary
    # ------------------------------------------------------------------
    for K in K_list:
        mu1 = np.mean(results[1][K]['test']) if 1 in results else float('nan')
        mu2 = np.mean(results[2][K]['test']) if 2 in results else float('nan')
        mu_all = np.mean(results['all'][K]['test'])
        qm_marker = ' <- QM' if K == 15 else ''
        print(f'{K:>4}  {mu1:>14.4f}  {mu2:>14.4f}  {mu_all:>14.4f}{qm_marker}')

    return results


# Expose default reg constant
ALS_REG_DEFAULT: float = 1e-6
