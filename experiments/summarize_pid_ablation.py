"""Utility for summarizing PID, vanilla-Lagrangian, and no-Lagrangian runs from figure/pid logs."""

import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
import numpy as np
import _bootstrap  # noqa: F401
from pid_saca.paths import REPO_ROOT

BASEFILE = REPO_ROOT


class Config:

    def __init__(self):

        self.figure_dir = BASEFILE / "figure" / "pid"
        self.pattern = "pid_saca_training_log_*.csv"


        self.avg_last_n = 100
        self.max_episodes = 200


        self.experiment_names = [
            "Vanilla Lagrangian",
            "PID Lagrangian",
            "No Lagrangian"
        ]


        self.metrics = ["revenue", "orr", "total_distance", "cost", "lambda"]


        self.output_filename = "pid_ablation_summary.csv"



def get_sorted_files(config: Config):


    search_path = config.figure_dir / config.pattern
    files = glob.glob(str(search_path))

    if not files:
        print(f"Error: No files found matching pattern: {search_path}")
        return []


    files_sorted = sorted(files, key=os.path.getmtime)
    return files_sorted



def analyze_results(config: Config):


    print("\n" + "=" * 80)
    print(f"PID Ablation Study Analysis (Average over last {config.avg_last_n} episodes)")
    print("=" * 80)


    files = get_sorted_files(config)

    if len(files) < len(config.experiment_names):
        print(f"Warning: Expected {len(config.experiment_names)} files, but found {len(files)}")


    num_experiments = len(config.experiment_names)
    if len(files) > num_experiments:
        files = files[-num_experiments:]

    print(f"\nFound {len(files)} files in {config.figure_dir}:")
    for i, f in enumerate(files):
        mtime = os.path.getmtime(f)
        mtime_str = pd.Timestamp.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        print(f"  [{i+1}] {os.path.basename(f)} (Modified: {mtime_str})")

    results = {}

    for i, (exp_name, file_path) in enumerate(zip(config.experiment_names, files)):
        try:
            df = pd.read_csv(file_path)


            if config.max_episodes and len(df) > config.max_episodes:
                df = df.iloc[:config.max_episodes]

            if len(df) == 0:
                print(f"Warning: Empty data for {exp_name}")
                continue


            last_n = df.tail(config.avg_last_n)
            exp_res = {}

            for m in config.metrics:
                if m in df.columns:
                    if m == "lambda":

                        exp_res[m] = df[m].var()
                    else:

                        exp_res[m] = last_n[m].mean()
                else:
                    exp_res[m] = float('nan')
                    print(f"  Warning: Metric '{m}' not found in {os.path.basename(file_path)}")

            results[exp_name] = exp_res
            print(f"\nLoaded: {exp_name} <- {os.path.basename(file_path)}")
            print(f"  Episodes: {len(df)}, Analyzing last {min(config.avg_last_n, len(df))} episodes")

        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")


    if not results:
        print("No results to display.")
        return None


    df_summary = pd.DataFrame(results).T


    available_metrics = [m for m in config.metrics if m in df_summary.columns]
    df_summary = df_summary[available_metrics]


    column_rename = {
        "revenue": "Profit",
        "orr": "ORR",
        "total_distance": "Dist.(km)",
        "cost": "Cost",
        "lambda": "Lambda Var."
    }
    df_summary = df_summary.rename(columns=column_rename)


    print("\n" + "=" * 80)
    print("Summary Table:")
    print("=" * 80)
    pd.set_option('display.float_format', lambda x: '%.4f' % x)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(df_summary.to_string())
    print("=" * 80)


    output_path = config.figure_dir / config.output_filename
    print(f"\nSummary table saved to: {output_path}")

    return df_summary



if __name__ == "__main__":

    config = Config()


    analyze_results(config)
