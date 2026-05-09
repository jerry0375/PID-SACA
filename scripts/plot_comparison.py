"""Plotting utility for comparison curves stored under figure/compare."""

import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
import _bootstrap  # noqa: F401
from pid_saca.paths import FIGURE_DIR


plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 12,
    'axes.labelsize': 55,
    'axes.titlesize': 16,
    'legend.fontsize': 35,
    'xtick.labelsize': 55,
    'ytick.labelsize': 55
})

def plot_comparison():
    log_dir2 = FIGURE_DIR / "compare"
    log_dir = log_dir2
    max_episodes = 150


    SMOOTHING_ALPHA = 0.2


    models_config = [

        ("pid_saca_training_log_*.csv", "PID-SACA", "#D62728", "-"),


        ("grc_elg_training_log_*.csv", "GRC-ELG", "#1F77B4", "-"),


        ("saca_training_log_*.csv", "SACA", "#FF7F0E", "-"),


        ("jdrl_training_log_*.csv", "JDRL-POMO", "#2CA02C", "-"),


        ("mlp_training_log_*.csv", "ST-SACA", "#7F7F7F", "--")
    ]

    data_to_plot = []

    print("Searching for log files in:", log_dir)

    for pattern, label, color, linestyle in models_config:
        search_path = os.path.join(log_dir, pattern)
        files = glob.glob(search_path)

        if not files:
            print(f"Warning: No files found for pattern '{pattern}'")
            continue


        latest_file = sorted(files)[-1]
        print(f"Found for {label}: {latest_file}")

        try:
            df = pd.read_csv(latest_file)


            df = df.iloc[:max_episodes]

            data_to_plot.append({
                "df": df,
                "label": label,
                "color": color,
                "linestyle": linestyle
            })
        except Exception as e:
            print(f"Error reading {latest_file}: {e}")

    if not data_to_plot:
        print("No valid data found to plot.")
        return


    metrics = ["total_reward", "revenue", "cost", "orr"]
    summary_metrics = ["total_reward", "revenue", "cost", "orr", "total_distance"]
    ylabels = ["Reward", "Profit", "Cost", "ORR"]
    titles = ["Average Reward", "Average Revenue", "Average Cost", "Average ORR"]


    print("\n" + "="*60)
    print(f"Performance Comparison (Average over last 40 episodes of the first {max_episodes})")
    print("Baseline for %: Worst performing model = 100%")
    print("="*60)

    summary_data = {}

    for metric in summary_metrics:

        model_means = {}
        for item in data_to_plot:
            df = item["df"]
            if metric in df.columns:

                model_means[item["label"]] = df[metric].tail(40).mean()
            else:
                model_means[item["label"]] = float('nan')


        valid_values = [v for v in model_means.values() if not pd.isna(v)]
        if not valid_values:
            continue

        if metric in ["cost", "total_distance"]:

            baseline_value = max(valid_values)
        else:

            baseline_value = min(valid_values)


        metric_column = []
        for item in data_to_plot:
            label = item["label"]
            val = model_means[label]
            if pd.isna(val):
                metric_column.append("N/A")
            else:
                if baseline_value == 0:
                    percentage = 0.0
                else:
                    percentage = (val / baseline_value) * 100
                metric_column.append(f"{val:.2f} ({percentage:.1f}%)")

        summary_data[metric] = metric_column


    model_names = [item["label"] for item in data_to_plot]
    summary_df = pd.DataFrame(summary_data, index=model_names)
    print(summary_df)
    print("="*60 + "\n")

    plt.figure(figsize=(15, 10))

    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)

        for item in data_to_plot:
            df = item["df"]
            if metric in df.columns:


                smoothed_data = df[metric].ewm(alpha=SMOOTHING_ALPHA, adjust=False).mean()


                plt.plot(df["Episode"], smoothed_data,
                         label=item["label"],
                         color=item["color"],
                         linestyle=item["linestyle"],
                         linewidth=2.5)

            else:
                print(f"Metric '{metric}' not found for {item['label']}")

        plt.xlabel("Episode")
        plt.ylabel(ylabels[i])

        plt.legend(frameon=True, loc='best', edgecolor='black', fancybox=False)

        plt.grid(True, which='major', linestyle='--', alpha=0.4)

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


    plt.tight_layout()


    output_filename = "comparison_result_smoothed.png"
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"comparison_result_smoothed_{timestamp}.pdf"
    output_path = os.path.join(log_dir, output_filename)

    print(f"\nComparison plot saved to '{output_path}'")


    print("\nSaving separate PDF plots for each metric...")
    for i, metric in enumerate(metrics):
        fig_single, ax_single = plt.subplots(figsize=(16, 9))

        for item in data_to_plot:
            df = item["df"]
            if metric in df.columns:
                smoothed_data = df[metric].ewm(alpha=SMOOTHING_ALPHA, adjust=False).mean()
                ax_single.plot(df["Episode"], smoothed_data,
                         label=item["label"],
                         color=item["color"],
                         linestyle=item["linestyle"],
                         linewidth=4.5)

        ax_single.set_xlabel("Episode")
        ax_single.set_ylabel(ylabels[i])
        ax_single.legend(frameon=True, loc='best', edgecolor='black', fancybox=False)
        ax_single.grid(True, which='major', linestyle='--', alpha=0.4)

        ax_single.spines['top'].set_visible(True)
        ax_single.spines['right'].set_visible(True)
        ax_single.spines['bottom'].set_visible(True)
        ax_single.spines['left'].set_visible(True)

        fig_single.tight_layout()

        single_output_filename = f"comparison_{ylabels[i].lower()}.pdf"
        single_output_path = os.path.join(log_dir2, single_output_filename)
        fig_single.savefig(single_output_path, format='pdf', bbox_inches='tight')
        print(f"Saved '{single_output_path}'")
        plt.close(fig_single)


    plt.show()

if __name__ == "__main__":
    plot_comparison()
