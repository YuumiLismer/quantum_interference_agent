"""
simulator/interferometer.py
============================
Gaussian-beam 4-slit interferometer simulator.

Physical Model
--------------
Each slit k contributes a complex field amplitude at pixel position x:

    psi_k(x) = A_k * exp(-2*(x - x_k)**2 / r**2) * exp(1j*(phi_k + kappa_k*x))

where:
    A_k      in {0, 1}         -- shutter open (1) or closed (0)
    x_k                        -- slit centre position
    r = BEAM_RADIUS            -- Gaussian beam radius
    phi_k                      -- applied phase shift
    kappa_k  = KX_LIST[k]      -- transverse wave vector (tilt per slit)

The intensity (probability) at pixel x is:

    I(x) = |psi_0(x) + psi_1(x) + psi_2(x) + psi_3(x)|**2

After row-wise normalisation: p(x | setting s) = I(x) / sum_x I(x).

Measurement Settings
--------------------
Settings are organised by n_open (number of open slits):

    n_open=1: 4 shutter patterns  x 3 phase combinations  =  12 rows
    n_open=2: 6 shutter patterns  x 9 phase combinations  =  54 rows
    n_open=3: 4 shutter patterns  x 27 phase combinations = 108 rows
    n_open=4: 1 shutter pattern   x 81 phase combinations =  81 rows
    Total: 255 rows

GPT Rank Prediction
-------------------
For a 4-slit experiment, QM predicts GPT rank = n^2 - 1 = 15.
This follows from: density matrix rho in su(4) has dim(su(4)) = 15,
and Born rule p(x|rho) = Tr(E_x rho) is linear in rho.
"""

from itertools import combinations, product
import numpy as np

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
BEAM_RADIUS: float = 1.0
SLIT_X = np.array([-0.15, -0.05, 0.05, 0.15])
KX_LIST = np.array([-20., -10., 10., 20.])
NUM_PIXELS: int = 500
X_GRID = np.linspace(-1.0, 1.0, NUM_PIXELS)
RANDOM_SEED: int = 42

N_SLITS: int = 4

# ---------------------------------------------------------------------------
# Phase / shutter parameter space
# ---------------------------------------------------------------------------
PHASE_VALUES = [0.0, np.pi / 4, np.pi / 2]

# All 3^4 = 81 phase 4-tuples
PHASE_PATTERNS: list = list(product(PHASE_VALUES, repeat=N_SLITS))

# Human-readable shutter labels (O=open, X=closed)
# Generated from all 2^4 = 16 shutter patterns; we list them in n_open order.
_ALL_SHUTTER_COMBOS = []
for _n in range(1, N_SLITS + 1):
    for _combo in combinations(range(N_SLITS), _n):
        _label_parts = ['X', 'X', 'X', 'X']
        for _idx in _combo:
            _label_parts[_idx] = 'O'
        _ALL_SHUTTER_COMBOS.append((''.join(_label_parts), _combo))

SHUTTER_LABELS: list = [lbl for lbl, _ in _ALL_SHUTTER_COMBOS]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _amplitudes(label: str) -> tuple:
    """Parse a shutter label like 'OXOX' and return a tuple of amplitude values.

    Parameters
    ----------
    label : str
        4-character string where 'O' = open (amplitude 1) and 'X' = closed (0).

    Returns
    -------
    tuple of float
        (A_0, A_1, A_2, A_3) with values in {0.0, 1.0}.
    """
    return tuple(1.0 if c == 'O' else 0.0 for c in label)


def _phase_label(combo: tuple) -> str:
    """Format a phase 4-tuple as a human-readable string.

    Parameters
    ----------
    combo : tuple of float
        Four phase values in radians.

    Returns
    -------
    str
        E.g. '(0, pi/4, pi/2, 0)'.
    """
    def _fmt(v: float) -> str:
        if abs(v) < 1e-9:
            return '0'
        elif abs(v - np.pi / 4) < 1e-9:
            return 'pi/4'
        elif abs(v - np.pi / 2) < 1e-9:
            return 'pi/2'
        else:
            return f'{v:.3f}'
    return '(' + ', '.join(_fmt(v) for v in combo) + ')'


def _intensity_row(
    amplitudes: tuple,
    phases: tuple,
) -> np.ndarray:
    """Compute the normalised intensity profile for one measurement setting.

    Parameters
    ----------
    amplitudes : tuple of float
        (A_0, A_1, A_2, A_3) shutter amplitudes.
    phases : tuple of float
        (phi_0, phi_1, phi_2, phi_3) applied phase shifts in radians.

    Returns
    -------
    np.ndarray, shape (NUM_PIXELS,)
        Normalised probability distribution p(x | setting).
        Sums to 1.0 over pixels.
    """
    field = np.zeros(NUM_PIXELS, dtype=complex)
    for k in range(N_SLITS):
        A_k = amplitudes[k]
        if A_k == 0.0:
            continue
        phi_k = phases[k]
        # Gaussian envelope centred at x_k with radius BEAM_RADIUS
        envelope = A_k * np.exp(-2.0 * (X_GRID - SLIT_X[k]) ** 2 / BEAM_RADIUS ** 2)
        # Complex phase: applied phase + tilt
        phasor = np.exp(1j * (phi_k + KX_LIST[k] * X_GRID))
        field += envelope * phasor

    intensity = np.abs(field) ** 2
    total = intensity.sum()
    if total < 1e-15:
        # Degenerate case: all slits closed or zero overlap
        return np.ones(NUM_PIXELS, dtype=float) / NUM_PIXELS
    return intensity / total


# ---------------------------------------------------------------------------
# Main simulator class
# ---------------------------------------------------------------------------

class InterferometerSimulator:
    """Simulate a 4-slit Gaussian-beam interferometer.

    Generates the probability matrix D[s, x] = p(pixel x | setting s)
    for all 255 measurement settings (shutter + phase combinations),
    grouped by n_open (number of open slits).

    Parameters
    ----------
    n_eff : float
        Effective photon count per measurement setting (used for Poisson
        noise simulation). Default: 50 000.
    seed : int
        Random seed for reproducibility. Default: 42.

    Attributes
    ----------
    n_eff : float
    rng : np.random.Generator
    """

    def __init__(self, n_eff: float = 50_000, seed: int = RANDOM_SEED):
        self.n_eff = float(n_eff)
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Internal: compute exact (noiseless) probability matrices
    # ------------------------------------------------------------------

    def _exact_probs(self, n_open: int) -> tuple:
        """Compute exact probability matrices for all settings with n_open slits open.

        Parameters
        ----------
        n_open : int
            Number of open slits (1, 2, 3, or 4).

        Returns
        -------
        matrix : np.ndarray, shape (n_rows, NUM_PIXELS)
            Exact (noiseless) probability distributions.
        labels : list of str
            Human-readable label for each row (shutter pattern + phase).
        """
        rows = []
        labels = []

        # Iterate over all shutter patterns with exactly n_open slits open
        for combo in combinations(range(N_SLITS), n_open):
            # Build amplitude vector for this shutter pattern
            A = [0.0] * N_SLITS
            for idx in combo:
                A[idx] = 1.0
            A_tuple = tuple(A)

            # Build shutter label string
            shutter_str = ''.join('O' if A[k] else 'X' for k in range(N_SLITS))

            # Iterate over all phase patterns for the *open* slits only
            # (n_open^? where ? = n_open, cycling through PHASE_VALUES)
            open_phase_combos = list(product(PHASE_VALUES, repeat=n_open))
            for open_phases in open_phase_combos:
                # Assign phases: open slits get the phase, closed slits get 0
                phases = [0.0] * N_SLITS
                for j, idx in enumerate(combo):
                    phases[idx] = open_phases[j]
                phases_tuple = tuple(phases)

                row = _intensity_row(A_tuple, phases_tuple)
                rows.append(row)
                labels.append(f'{shutter_str}_{_phase_label(phases_tuple)}')

        matrix = np.array(rows, dtype=float)  # shape: (n_rows, NUM_PIXELS)
        return matrix, labels

    # ------------------------------------------------------------------
    # Public: simulate full dataset
    # ------------------------------------------------------------------

    def simulate(self, noise: bool = True) -> dict:
        """Simulate the full interferometer dataset.

        Computes the probability matrix for all n_open groups, with optional
        Poisson noise. The noisy matrix is obtained by:
            counts = rng.poisson(exact * n_eff)
            noisy  = counts / counts.sum(axis=1, keepdims=True)

        Parameters
        ----------
        noise : bool
            If True, add Poisson noise. If False, return exact probabilities.

        Returns
        -------
        dict
            Keys are n_open values (1, 2, 3, 4). Each value is a dict:
                'matrix'  : np.ndarray, shape (n_rows, NUM_PIXELS)
                            Noisy (or exact) probability distributions.
                'exact'   : np.ndarray, shape (n_rows, NUM_PIXELS)
                            Exact (noiseless) probability distributions.
                'labels'  : list of str
                'n_eff'   : float
        """
        result = {}
        for n_open in range(1, N_SLITS + 1):
            exact, labels = self._exact_probs(n_open)

            if noise:
                # Poisson-sample each row independently
                counts = self.rng.poisson(exact * self.n_eff)  # (n_rows, NUM_PIXELS)
                row_sums = counts.sum(axis=1, keepdims=True)
                # Protect against zero rows (very rare at reasonable n_eff)
                row_sums = np.where(row_sums == 0, 1, row_sums)
                noisy = counts.astype(float) / row_sums
            else:
                noisy = exact.copy()

            result[n_open] = {
                'matrix': noisy,
                'exact':  exact,
                'labels': labels,
                'n_eff':  self.n_eff,
            }

        return result
