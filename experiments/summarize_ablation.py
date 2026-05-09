"""Utility for summarizing ablation logs from the local log directory."""

import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
import numpy as np
import _bootstrap  # noqa: F401
from pid_saca.paths import LOG_DIR as ROOT_LOG_DIR


LOG_DIR = str(ROOT_LOG_DIR)
MAX_EPISODES = 200


EXPERIMENTS = [
    {
        "name": "PID-SACA (Full)",
        "pattern": "pid_saca_training_log_*.csv"
    },
    {
        "name": "w/o Spatial Encoding",
        "pattern": "saca_training_log_*.csv"
    },
    {
        "name": "w/o Route Model",
        "pattern": "ablation_log_wo_AttnRoute_*.csv"
    },
    {
        "name": "w/o Dynamic Pricing",
        "pattern": "ablation_wo_pricing_training_log_*.csv"
    }
]


def get_latest_file(pattern):

    search_path = os.path.join(LOG_DIR, pattern)
    files = glob.glob(search_path)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def analyze_results():


    print("\n" + "="*80)
    print("Ablation Study Analysis (Average over last 10 episodes)")
    print("="*80)

    metrics = ["total_reward", "revenue", "cost", "orr", "total_distance"]
    results = {}

    for exp in EXPERIMENTS:

        pattern = exp['pattern']
        latest_file = get_latest_file(pattern)

        if not latest_file:
            print(f"Warning: No data found for {exp['name']} (Pattern: {pattern})")
            continue

        try:
            df = pd.read_csv(latest_file)


            if len(df) > MAX_EPISODES:
                df = df.iloc[:MAX_EPISODES]

            if len(df) == 0:
                print(f"Warning: Empty data for {exp['name']}")
                continue


            last_10 = df.tail(10)
            exp_res = {}
            for m in metrics:
                if m in df.columns:
                    exp_res[m] = last_10[m].mean()
                else:
                    exp_res[m] = float('nan')

            results[exp['name']] = exp_res
            print(f"Loaded data for {exp['name']} from {os.path.basename(latest_file)}")

        except Exception as e:
            print(f"Error analyzing {latest_file}: {e}")


    if not results:
        print("No results to display.")
        return


    df_summary = pd.DataFrame(results).T


    available_metrics = [m for m in metrics if m in df_summary.columns]
    df_summary = df_summary[available_metrics]


    pd.set_option('display.float_format', lambda x: '%.4f' % x)
    print("\nSummary Table:")
    print(df_summary)


    output_path = os.path.join(LOG_DIR, "ablation_study_summary.csv")
    df_summary.to_csv(output_path)
    print(f"\nSummary table saved to {output_path}")


if __name__ == "__main__":

    analyze_results()
