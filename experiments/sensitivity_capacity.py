"""Capacity sensitivity helper. The analysis path expects one log per capacity value in figure/demand."""



import os
import glob
import pandas as pd
import numpy as np
import pathlib


import _bootstrap  # noqa: F401
from pid_saca.algorithms import pid_saca as PID_SACA
from pid_saca.algorithms import grc_elg as GRC_ELG
from pid_saca.algorithms import jdrl_pomo as JDRL_POMO
from pid_saca.algorithms import saca as SACA
from pid_saca.algorithms import saca_baseline as SACA_MLP
from pid_saca.paths import REPO_ROOT

BASEFILE = REPO_ROOT


class DemandParaTestConfig:


    def __init__(self):

        self.capacity_values = [20, 40, 60, 80]


        self.episodes = 150


        self.avg_last_n = 40


        self.log_dir = BASEFILE / "log"
        self.figure_dir = BASEFILE / "figure" / "demand"


        self.algorithms = [
            {
                "name": "PID-SACA",
                "module": PID_SACA,
                "train_func": "train_saca",
                "pattern": "pid_saca_training_log_*.csv"
            },
            {
                "name": "GRC-ELG",
                "module": GRC_ELG,
                "train_func": "train_grc_elg",
                "pattern": "grc_elg_training_log_*.csv"
            },
            {
                "name": "JDRL-POMO",
                "module": JDRL_POMO,
                "train_func": "train_jdrl",
                "pattern": "jdrl_training_log_*.csv"
            },
            {
                "name": "ST-SACA",
                "module": SACA,
                "train_func": "train_saca",
                "pattern": "saca_training_log_*.csv"
            },
            {
                "name": "SACA-MLP",
                "module": SACA_MLP,
                "train_func": "train_saca",
                "pattern": "mlp_training_log_*.csv"
            }
        ]



def run_demand_experiments(config: DemandParaTestConfig):


    print("=" * 70)
    print("Starting Demand Parameter (Bus Capacity) Sensitivity Experiments")
    print(f"Capacity values to test: {config.capacity_values}")
    print(f"Algorithms to compare: {[alg['name'] for alg in config.algorithms]}")
    print(f"Episodes per experiment: {config.episodes}")
    print("=" * 70)

    total_experiments = len(config.capacity_values) * len(config.algorithms)
    current_exp = 0

    for capacity in config.capacity_values:
        print(f"\n{'='*70}")
        print(f"Testing Bus Capacity = {capacity}")
        print(f"{'='*70}")

        for alg in config.algorithms:
            current_exp += 1
            print(f"\n>>> [{current_exp}/{total_experiments}] Running {alg['name']} with capacity={capacity}")

            try:

                module = alg["module"]
                train_func = getattr(module, alg["train_func"])


                train_cfg = module.Config()
                train_cfg.bus_capacity = capacity


                if hasattr(train_cfg, 'episodes'):
                    train_cfg.episodes = config.episodes
                if hasattr(train_cfg, 'max_episodes'):
                    train_cfg.max_episodes = config.episodes
                if hasattr(train_cfg, 'min_episodes'):
                    train_cfg.min_episodes = config.episodes


                train_func(train_cfg)
                print(f"    {alg['name']} with capacity={capacity} completed successfully.")

            except Exception as e:
                print(f"    [Error] {alg['name']} with capacity={capacity} failed: {e}")
                import traceback
                traceback.print_exc()

    print("\n" + "=" * 70)
    print("All Demand Parameter Experiments Completed!")
    print(f"Log files saved to: {config.log_dir}")
    print("=" * 70)



def plot_demand_results(config: DemandParaTestConfig):


    print("\n>>> Analyzing Demand Parameter Sensitivity Results...")

    num_capacities = len(config.capacity_values)
    num_algorithms = len(config.algorithms)


    metrics = ['Profit', 'ORR', 'Cost', 'Dist.(km)']


    all_results = {}

    for alg in config.algorithms:
        alg_name = alg["name"]
        pattern = alg["pattern"]
        search_path = config.figure_dir / pattern
        files = glob.glob(str(search_path))

        if len(files) < num_capacities:
            print(f"Warning: Expected {num_capacities} files for {alg_name}, but found {len(files)}")
            if len(files) == 0:
                print(f"  No files found for pattern: {search_path}")
                all_results[alg_name] = None
                continue


        files_sorted = sorted(files, key=os.path.getmtime)
        latest_files = files_sorted[-num_capacities:]

        print(f"\n{alg_name}: Found {len(latest_files)} files")

        alg_results = {}
        for i, (capacity, file_path) in enumerate(zip(config.capacity_values, latest_files)):
            try:
                df = pd.read_csv(file_path)
                last_n = df.tail(config.avg_last_n)

                avg_profit = last_n['revenue'].mean() if 'revenue' in df.columns else np.nan
                avg_orr = last_n['orr'].mean() if 'orr' in df.columns else np.nan
                avg_cost = last_n['cost'].mean() if 'cost' in df.columns else np.nan
                avg_dist = last_n['total_distance'].mean() if 'total_distance' in df.columns else np.nan

                alg_results[capacity] = {
                    'Profit': avg_profit,
                    'ORR': avg_orr,
                    'Cost': avg_cost,
                    'Dist.(km)': avg_dist
                }

                print(f"  Capacity={capacity}: Profit={avg_profit:.4f}, ORR={avg_orr:.4f}, Cost={avg_cost:.4f}, Dist={avg_dist:.4f} ({os.path.basename(file_path)})")

            except Exception as e:
                print(f"  [Error] Failed to read {file_path}: {e}")
                alg_results[capacity] = {m: np.nan for m in metrics}

        all_results[alg_name] = alg_results


    row_tuples = []
    for capacity in config.capacity_values:
        for metric in metrics:
            row_tuples.append((capacity, metric))

    row_index = pd.MultiIndex.from_tuples(row_tuples, names=['Capacity', 'Metric'])


    data = {}
    for alg in config.algorithms:
        alg_name = alg["name"]
        if all_results.get(alg_name) is None:
            data[alg_name] = [np.nan] * len(row_tuples)
        else:
            col_data = []
            for capacity in config.capacity_values:
                if capacity in all_results[alg_name]:
                    for metric in metrics:
                        col_data.append(all_results[alg_name][capacity][metric])
                else:
                    col_data.extend([np.nan] * len(metrics))
            data[alg_name] = col_data

    df_results = pd.DataFrame(data, index=row_index)


    print("\n" + "=" * 80)
    print(f"Demand Parameter Sensitivity Results (Average over last {config.avg_last_n} episodes)")
    print("=" * 80)


    pd.set_option('display.float_format', lambda x: f'{x:.4f}')
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(df_results.to_string())
    print("=" * 80)


    config.figure_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.figure_dir / "demand_sensitivity_results.csv"
    df_results.to_csv(output_path)
    print(f"\nResults saved to: {output_path}")

    return df_results



if __name__ == "__main__":

    config = DemandParaTestConfig()




    plot_demand_results(config)
