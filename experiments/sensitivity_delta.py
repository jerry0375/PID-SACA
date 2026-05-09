"""Target-ORR sensitivity helper. The analysis path maps existing logs to delta values by file order."""



import os
import glob
import pandas as pd
import numpy as np
import pathlib
import _bootstrap  # noqa: F401
from pid_saca.algorithms import pid_saca as core
from pid_saca.paths import REPO_ROOT

BASEFILE = REPO_ROOT


class DeltaTestConfig:


    def __init__(self):

        self.delta_values = [0.10, 0.25, 0.40, 0.55, 0.70, 0.80, 0.90, 1.00]


        self.episodes = 200


        self.avg_last_n = 10


        self.log_dir = BASEFILE / "log"
        self.figure_dir = BASEFILE / "figure" / "delta"
        self.log_pattern = "pid_saca_training_log_*.csv"



def run_delta_experiments(config: DeltaTestConfig):


    print("=" * 60)
    print("Starting Delta Sensitivity Experiments")
    print(f"Delta values to test: {config.delta_values}")
    print(f"Episodes per experiment: {config.episodes}")
    print("=" * 60)

    for i, delta in enumerate(config.delta_values):
        print(f"\n>>> [{i+1}/{len(config.delta_values)}] Running experiment with delta (target_orr) = {delta}")


        train_cfg = core.Config()
        train_cfg.target_orr = delta
        train_cfg.episodes = config.episodes
        train_cfg.max_episodes = config.episodes


        try:
            core.train_saca(train_cfg)
            print(f"    Experiment with delta={delta} completed successfully.")
        except Exception as e:
            print(f"    [Error] Experiment with delta={delta} failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("All Delta Sensitivity Experiments Completed!")
    print(f"Log files saved to: {config.log_dir}")
    print("=" * 60)



def plot_delta_results(config: DeltaTestConfig):


    print("\n>>> Analyzing Delta Sensitivity Results...")


    search_path = config.figure_dir / config.log_pattern
    files = glob.glob(str(search_path))

    num_deltas = len(config.delta_values)

    if len(files) < num_deltas:
        print(f"Warning: Expected {num_deltas} log files, but found {len(files)}")
        print(f"Search path: {search_path}")
        if len(files) == 0:
            print("No files found. Please run experiments first or copy log files to figure/delta/")
            return None


    files_sorted = sorted(files, key=os.path.getmtime)
    latest_files = files_sorted[-num_deltas:]

    print(f"Found {len(latest_files)} log files:")
    for f in latest_files:
        print(f"  - {os.path.basename(f)}")


    results = []

    for i, (delta, file_path) in enumerate(zip(config.delta_values, latest_files)):
        try:
            df = pd.read_csv(file_path)


            last_n = df.tail(config.avg_last_n)


            avg_profit = last_n['revenue'].mean() if 'revenue' in df.columns else np.nan
            avg_orr = last_n['orr'].mean() if 'orr' in df.columns else np.nan

            results.append({
                'delta': delta,
                'Profit': avg_profit,
                'ORR': avg_orr
            })

            print(f"  Delta={delta}: Profit={avg_profit:.4f}, ORR={avg_orr:.4f}")

        except Exception as e:
            print(f"  [Error] Failed to read {file_path}: {e}")
            results.append({
                'delta': delta,
                'Profit': np.nan,
                'ORR': np.nan
            })


    df_results = pd.DataFrame(results)
    df_results = df_results.set_index('delta')


    df_transposed = df_results.T

    print("\n" + "=" * 60)
    print(f"Delta Sensitivity Results (Average over last {config.avg_last_n} episodes)")
    print("=" * 60)
    print(df_transposed.to_string())
    print("=" * 60)


    output_path = config.figure_dir / "delta_sensitivity_results.csv"
    config.figure_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nResults saved to: {output_path}")

    return df_transposed



if __name__ == "__main__":

    config = DeltaTestConfig()




    plot_delta_results(config)
