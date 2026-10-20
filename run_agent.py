#!/usr/bin/env python3
"""
run_agent.py
============
Entry point for the Quantum Interference Research Scientist Agent.

Usage
-----
    python run_agent.py                           # N_eff=50000, noisy
    python run_agent.py --n_eff 1000 10000 50000  # sweep noise levels
    python run_agent.py --no-noise                # noiseless theory data
    python run_agent.py --k_max 20 --n_folds 10   # custom sweep params
    python run_agent.py --output results/         # custom output directory
"""

import argparse
from agent.scientist import ResearchScientist


def main():
    parser = argparse.ArgumentParser(
        description='Quantum Interference GPT Rank — Research Scientist Agent',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--n_eff', type=float, nargs='+', default=[50_000],
                        help='Effective photon counts to simulate (one or more).')
    parser.add_argument('--no-noise', action='store_true',
                        help='Use noiseless theory data.')
    parser.add_argument('--k_max', type=int, default=20,
                        help='Maximum rank K to sweep.')
    parser.add_argument('--n_folds', type=int, default=10,
                        help='Cross-validation folds.')
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help='Parallel jobs for CV sweep (-1 = all cores).')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory for REPORT.md and results.json.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed.')
    args = parser.parse_args()

    agent = ResearchScientist(
        n_eff_list = args.n_eff,
        noise      = not args.no_noise,
        k_max      = args.k_max,
        n_folds    = args.n_folds,
        n_jobs     = args.n_jobs,
        output_dir = args.output,
        seed       = args.seed,
    )
    agent.run()


if __name__ == '__main__':
    main()
