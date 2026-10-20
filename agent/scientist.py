"""
agent/scientist.py
==================
ResearchScientist: an autonomous 6-step scientific workflow agent for
GPT rank estimation in a 4-slit quantum interferometer.

Scientific Question
-------------------
Does the probability matrix D[s, x] = p(pixel x | setting s) of a 4-slit
interferometer have GPT rank 15 (quantum mechanics), rank 4 (classical),
or some intermediate value?

6-Step Workflow
---------------
1. formulate_hypothesis  -- state H0 (QM), H1 (classical), H2 (2nd-order)
2. design_experiment     -- document the 4-slit measurement settings
3. simulate_data         -- generate D using the Gaussian beam model
4. train_and_evaluate    -- ALS rank sweep with 10-fold cross-validation
5. analyse_results       -- find best K, compute delta-chi2, deliver verdict
6. write_report          -- emit REPORT.md and results.json
"""

import json
import os
import time
from datetime import datetime

import numpy as np

from simulator.interferometer import InterferometerSimulator
from model.rank_sweep import run_rank_sweep
from agent.reporter import ReportWriter


class ResearchScientist:
    """Autonomous research scientist agent for GPT rank estimation.

    Parameters
    ----------
    n_eff_list : list of float
        Effective photon count(s) to simulate. One full sweep is run per value.
    noise : bool
        Whether to add Poisson noise to the simulated data.
    k_max : int
        Maximum rank K to sweep (sweep is range(1, k_max+1)).
    n_folds : int
        Number of cross-validation folds.
    n_jobs : int
        Number of parallel workers for joblib. -1 = all cores.
    output_dir : str
        Directory for REPORT.md and results.json.
    seed : int
        Master random seed.
    """

    # ------------------------------------------------------------------
    # Class-level scientific constants
    # ------------------------------------------------------------------
    QM_RANK: int = 15
    CLASSICAL_RANK: int = 4
    SECOND_ORDER: int = 10

    def __init__(
        self,
        n_eff_list=None,
        noise: bool = True,
        k_max: int = 20,
        n_folds: int = 10,
        n_jobs: int = -1,
        output_dir: str = 'results',
        seed: int = 42,
    ):
        self.n_eff_list = list(n_eff_list) if n_eff_list is not None else [50_000]
        self.noise = noise
        self.k_max = k_max
        self.n_folds = n_folds
        self.n_jobs = n_jobs
        self.output_dir = output_dir
        self.seed = seed
        self._log: list = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _say(self, msg: str) -> None:
        """Print a timestamped agent message and append it to the log."""
        ts = datetime.now().strftime('%H:%M:%S')
        line = f'[Agent {ts}] {msg}'
        print(line)
        self._log.append(line)

    # ------------------------------------------------------------------
    # Step 1: Formulate hypothesis
    # ------------------------------------------------------------------

    def formulate_hypothesis(self) -> dict:
        """State the scientific hypotheses and model selection criterion.

        Returns
        -------
        dict with keys:
            H0_QM        -- quantum mechanics hypothesis (rank=15)
            H1_classical -- classical hypothesis (rank=4)
            H2_2nd_order -- second-order interference hypothesis (rank~10)
            criterion    -- model selection criterion description
        """
        self._say('Step 1/6 -- Formulating hypothesis...')
        hypothesis = {
            'H0_QM': (
                f'Quantum mechanics: GPT rank = {self.QM_RANK} '
                f'(= n^2-1 = 4^2-1, from dim su(4))'
            ),
            'H1_classical': (
                f'Classical wave optics: GPT rank = {self.CLASSICAL_RANK} '
                f'(one real amplitude per slit)'
            ),
            'H2_2nd_order': (
                f'2nd-order interference only: GPT rank ~ {self.SECOND_ORDER} '
                f'(pairwise slit correlations only)'
            ),
            'criterion': (
                'Model selection: K* = argmin_K mean_folds[chi2_test(K)/dof]. '
                'Verdict: K*==15 supports QM, K*<=4 supports classical, '
                f'K* in (4,15) is intermediate.'
            ),
        }
        self._say(f'  H0 (QM):        rank = {self.QM_RANK}')
        self._say(f'  H1 (classical): rank = {self.CLASSICAL_RANK}')
        self._say(f'  H2 (2nd-order): rank ~ {self.SECOND_ORDER}')
        self._say(f'  Criterion: {hypothesis["criterion"]}')
        return hypothesis

    # ------------------------------------------------------------------
    # Step 2: Design experiment
    # ------------------------------------------------------------------

    def design_experiment(self) -> dict:
        """Document the experimental design parameters.

        Returns
        -------
        dict describing the full measurement setting space.
        """
        self._say('Step 2/6 -- Designing experiment...')
        design = {
            'n_slits': 4,
            'n_phases': 3,
            'phase_values': [0.0, 'pi/4', 'pi/2'],
            'n_phase_patterns': 81,
            'shutter_groups': {
                1: {'n_shutter_patterns': 4,  'n_phase_patterns': 3,   'total_rows': 12},
                2: {'n_shutter_patterns': 6,  'n_phase_patterns': 9,   'total_rows': 54},
                3: {'n_shutter_patterns': 4,  'n_phase_patterns': 27,  'total_rows': 108},
                4: {'n_shutter_patterns': 1,  'n_phase_patterns': 81,  'total_rows': 81},
            },
            'total_settings': 255,
            'n_pixels': 500,
            'n_eff_list': self.n_eff_list,
            'K_range': list(range(1, self.k_max + 1)),
            'n_folds': self.n_folds,
            'noise': self.noise,
            'fit_strategy': 'joint V across all n_open groups; ALS with SVD warm-start',
            'k_max': self.k_max,
        }
        total = sum(v['total_rows'] for v in design['shutter_groups'].values())
        self._say(f'  n_slits={design["n_slits"]}, n_phase_patterns={design["n_phase_patterns"]}')
        self._say(f'  Total measurement settings: {total}')
        self._say(f'  K sweep: 1..{self.k_max}, folds={self.n_folds}, noise={self.noise}')
        return design

    # ------------------------------------------------------------------
    # Step 3: Simulate data
    # ------------------------------------------------------------------

    def simulate_data(self) -> dict:
        """Generate simulated probability matrices for all n_eff values.

        Returns
        -------
        dict : {n_eff (float): data_dict}
            data_dict is the output of InterferometerSimulator.simulate().
        """
        self._say(
            f'Step 3/6 -- Simulating data '
            f'(n_eff={self.n_eff_list}, noise={self.noise})...'
        )
        datasets = {}
        for n_eff in self.n_eff_list:
            sim = InterferometerSimulator(n_eff=n_eff, seed=self.seed)
            data = sim.simulate(noise=self.noise)
            datasets[n_eff] = data
            for n_open in sorted(data.keys()):
                mat = data[n_open]['matrix']
                self._say(
                    f'  n_eff={n_eff:.0f}, n_open={n_open}: '
                    f'matrix shape {mat.shape}, '
                    f'row mean={mat.mean():.4e}'
                )
        return datasets

    # ------------------------------------------------------------------
    # Step 4: Train and evaluate
    # ------------------------------------------------------------------

    def train_and_evaluate(self, datasets: dict) -> dict:
        """Run ALS rank sweep with cross-validation for each n_eff dataset.

        Parameters
        ----------
        datasets : dict
            Output of simulate_data().

        Returns
        -------
        dict : {n_eff (float): sweep_results}
            sweep_results is the output of run_rank_sweep().
        """
        self._say('Step 4/6 -- Training and evaluating models...')
        all_sweep_results = {}
        for n_eff, data in datasets.items():
            self._say(f'  Running rank sweep for N_eff={n_eff:.0f}...')
            t0 = time.time()
            sweep = run_rank_sweep(
                data=data,
                K_range=range(1, self.k_max + 1),
                n_folds=self.n_folds,
                n_jobs=self.n_jobs,
                seed=self.seed,
            )
            elapsed = time.time() - t0
            self._say(f'  Sweep complete in {elapsed:.1f}s for N_eff={n_eff:.0f}')
            all_sweep_results[n_eff] = sweep
        return all_sweep_results

    # ------------------------------------------------------------------
    # Step 5: Analyse results
    # ------------------------------------------------------------------

    def analyse_results(self, sweep_results: dict, hypothesis: dict) -> dict:
        """Find optimal K per group, compute delta-chi2, deliver verdict.

        For each (n_eff, group) combination:
          - best_K = argmin(mean test chi2)
          - delta = mean_test_chi2(K=14) - mean_test_chi2(K=15)
          - consistent_with_QM = (best_K == QM_RANK)

        Parameters
        ----------
        sweep_results : dict
            Output of train_and_evaluate().
        hypothesis : dict
            Output of formulate_hypothesis().

        Returns
        -------
        dict : nested analysis results.
        """
        self._say('Step 5/6 -- Analysing results...')
        analysis = {}

        for n_eff, sweep in sweep_results.items():
            analysis[n_eff] = {}
            for group in sweep.keys():
                K_list = sorted(sweep[group].keys())
                test_means = [
                    np.mean(sweep[group][K]['test']) for K in K_list
                ]
                best_idx = int(np.argmin(test_means))
                best_K = K_list[best_idx]
                best_chi2 = test_means[best_idx]

                # Delta chi2: improvement from K=14 to K=15
                chi2_at_14 = np.mean(sweep[group][14]['test']) if 14 in sweep[group] else float('nan')
                chi2_at_15 = np.mean(sweep[group][15]['test']) if 15 in sweep[group] else float('nan')
                delta = chi2_at_14 - chi2_at_15

                consistent_with_QM = (best_K == self.QM_RANK)

                if best_K == self.QM_RANK:
                    verdict = f'SUPPORTS QM (rank={self.QM_RANK})'
                elif best_K <= self.CLASSICAL_RANK:
                    verdict = f'SUPPORTS CLASSICAL (rank<={self.CLASSICAL_RANK})'
                elif best_K <= self.SECOND_ORDER:
                    verdict = f'2ND-ORDER INTERFERENCE (rank={best_K})'
                else:
                    verdict = f'INTERMEDIATE rank={best_K} (K>{self.SECOND_ORDER})'

                analysis[n_eff][group] = {
                    'best_K': best_K,
                    'best_chi2': best_chi2,
                    'chi2_at_14': chi2_at_14,
                    'chi2_at_15': chi2_at_15,
                    'delta_chi2_14_to_15': delta,
                    'consistent_with_QM': consistent_with_QM,
                    'verdict': verdict,
                    'all_K': K_list,
                    'all_test_chi2': test_means,
                }

                self._say(
                    f'  N_eff={n_eff:.0f}, group={group}: '
                    f'best_K={best_K}, chi2={best_chi2:.4f}, '
                    f'delta(14->15)={delta:.4f} -> {verdict}'
                )

        return analysis

    # ------------------------------------------------------------------
    # Step 6: Write report
    # ------------------------------------------------------------------

    def write_report(
        self,
        hypothesis: dict,
        design: dict,
        analysis: dict,
        sweep_results: dict,
    ) -> None:
        """Write REPORT.md and results.json to output_dir.

        Parameters
        ----------
        hypothesis : dict
            From formulate_hypothesis().
        design : dict
            From design_experiment().
        analysis : dict
            From analyse_results().
        sweep_results : dict
            From train_and_evaluate().
        """
        self._say('Step 6/6 -- Writing report...')
        os.makedirs(self.output_dir, exist_ok=True)

        # -- REPORT.md --
        report_path = os.path.join(self.output_dir, 'REPORT.md')
        writer = ReportWriter(
            hypothesis=hypothesis,
            design=design,
            analysis=analysis,
            sweep_results=sweep_results,
            agent_log=self._log,
        )
        writer.write(report_path)
        self._say(f'  Wrote {report_path}')

        # -- results.json --
        def _to_serialisable(obj):
            """Recursively convert numpy types to Python natives."""
            if isinstance(obj, dict):
                return {str(k): _to_serialisable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_to_serialisable(v) for v in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
                return str(obj)
            else:
                return obj

        # Build a JSON-safe summary (exclude raw matrices from sweep_results)
        json_payload = {
            'hypothesis': hypothesis,
            'design': design,
            'analysis': _to_serialisable(analysis),
            'sweep_summary': {},
        }
        for n_eff, sweep in sweep_results.items():
            json_payload['sweep_summary'][str(n_eff)] = {}
            for group, group_data in sweep.items():
                json_payload['sweep_summary'][str(n_eff)][str(group)] = {}
                for K, fold_data in group_data.items():
                    json_payload['sweep_summary'][str(n_eff)][str(group)][str(K)] = {
                        'mean_train': float(np.mean(fold_data['train'])),
                        'std_train':  float(np.std(fold_data['train'])),
                        'mean_test':  float(np.mean(fold_data['test'])),
                        'std_test':   float(np.std(fold_data['test'])),
                    }

        json_path = os.path.join(self.output_dir, 'results.json')
        with open(json_path, 'w') as fh:
            json.dump(_to_serialisable(json_payload), fh, indent=2)
        self._say(f'  Wrote {json_path}')

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Execute the full 6-step workflow and print the final verdict table."""
        t_start = time.time()
        self._say('=' * 60)
        self._say('Quantum Interference GPT Rank -- Research Scientist Agent')
        self._say('=' * 60)

        hypothesis   = self.formulate_hypothesis()
        design       = self.design_experiment()
        datasets     = self.simulate_data()
        sweep_results = self.train_and_evaluate(datasets)
        analysis     = self.analyse_results(sweep_results, hypothesis)
        self.write_report(hypothesis, design, analysis, sweep_results)

        elapsed = time.time() - t_start

        # ------------------------------------------------------------------
        # Final verdict table
        # ------------------------------------------------------------------
        print()
        print('\u2554' + '\u2550' * 50 + '\u2557')
        print('\u2551' + ' FINAL VERDICT TABLE'.center(50) + '\u2551')
        print('\u2560' + '\u2550' * 50 + '\u2563')

        n_consistent = 0
        n_total = 0
        for n_eff in self.n_eff_list:
            grp_analysis = analysis.get(n_eff, {})
            all_result = grp_analysis.get('all', {})
            best_K = all_result.get('best_K', '?')
            is_qm = all_result.get('consistent_with_QM', False)
            symbol = '\u2713' if is_qm else '\u2717'
            n_total += 1
            if is_qm:
                n_consistent += 1
            line = f'  N_eff={n_eff:.0f} | Best K={best_K} | QM: {symbol}'
            print('\u2551' + line.ljust(50) + '\u2551')

        print('\u2560' + '\u2550' * 50 + '\u2563')
        summary = f'  {n_consistent}/{n_total} N_eff values consistent with QM rank=15'
        print('\u2551' + summary.ljust(50) + '\u2551')
        print('\u255a' + '\u2550' * 50 + '\u255d')
        print()
        print(f'Total runtime: {elapsed:.1f}s')
        print(f'Agent complete. See {self.output_dir}/REPORT.md')
