"""PID-vs-vanilla stability helper that trains or plots selected PID controller configurations."""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import _bootstrap  # noqa: F401
from pid_saca.algorithms import pid_saca as core
from pid_saca.paths import FIGURE_DIR, REPO_ROOT


class ExpConfig:


    def __init__(self):

        self.episodes = 200
        self.target_orr = 0.5
        self.seed = 42

        # --- Group A: Vanilla Lagrangian (Baseline) ---


        self.vanilla_kp = 0.0
        self.vanilla_ki = 0.5
        self.vanilla_kd = 0.0

        # --- Group B: PID Lagrangian (Ours) ---


        self.pid_kp = 5
        self.pid_ki = 1
        self.pid_kd = 5



class PlotConfig:


    def __init__(self):

        self.fig_width = 7
        self.fig_height = 6


        self.label_fontsize = 35
        self.tick_fontsize = 30
        self.legend_fontsize = 25


        self.linewidth = 4
        self.grid_alpha = 0.4
        self.grid_linestyle = '--'


        self.pid_color = '#D62728'
        self.vanilla_color = '#1F77B4'
        self.no_lag_color = '#2CA02C'


        self.save_dir = "figure/pid"


def run_training(exp_config):


    print("="*50)
    print("Starting Comparative Experiment: PID vs Vanilla")
    print("="*50)


    print("\n>>> Running Group A: Vanilla Lagrangian...")


    cfg_vanilla = core.Config()
    cfg_vanilla.episodes = exp_config.episodes
    cfg_vanilla.max_episodes = exp_config.episodes
    cfg_vanilla.target_orr = exp_config.target_orr


    cfg_vanilla.pid_kp = exp_config.vanilla_kp
    cfg_vanilla.pid_ki = exp_config.vanilla_ki
    cfg_vanilla.pid_kd = exp_config.vanilla_kd
    cfg_vanilla.pid_update_freq = 1


    core.train_saca(cfg_vanilla)


    print("\n>>> Running Group B: PID Lagrangian (Ours)...")

    cfg_pid = core.Config()
    cfg_pid.episodes = exp_config.episodes
    cfg_pid.max_episodes = exp_config.episodes
    cfg_pid.target_orr = exp_config.target_orr


    cfg_pid.pid_kp = exp_config.pid_kp
    cfg_pid.pid_ki = exp_config.pid_ki
    cfg_pid.pid_kd = exp_config.pid_kd
    cfg_pid.pid_update_freq = 1

    core.train_saca(cfg_pid)

    print("\nAll training sessions completed.")


def get_latest_three_log_files():

    log_dir = FIGURE_DIR / "pid"
    search_path = log_dir / "pid_saca_training_log_*.csv"
    files = glob.glob(str(search_path))
    if len(files) < 3:
        raise FileNotFoundError(f"Need at least 3 log files, but found {len(files)}")

    files_sorted = sorted(files, key=os.path.getmtime)


    return files_sorted[-3], files_sorted[-2], files_sorted[-1]

def plot_results(plot_cfg=None):


    if plot_cfg is None:
        plot_cfg = PlotConfig()

    print("\n>>> Generating Comparison Plots...")


    try:
        file_vanilla, file_pid, file_no_lag = get_latest_three_log_files()

        df_vanilla = pd.read_csv(file_vanilla)
        df_pid = pd.read_csv(file_pid)
        df_no_lag = pd.read_csv(file_no_lag)

        print(f"Loaded Vanilla Log:      {os.path.basename(file_vanilla)}")
        print(f"Loaded PID Log:          {os.path.basename(file_pid)}")
        print(f"Loaded No Lagrangian Log: {os.path.basename(file_no_lag)}")
    except Exception as e:
        print(f"Error loading logs: {e}")
        return

    # 2. Profit
    if 'revenue' in df_vanilla.columns:
        df_vanilla['profit'] = df_vanilla['revenue']
        df_pid['profit'] = df_pid['revenue']
        df_no_lag['profit'] = df_no_lag['revenue']
    else:
        print("Warning: 'revenue' column not found. Using 'total_reward' as proxy.")
        df_vanilla['profit'] = df_vanilla['total_reward']
        df_pid['profit'] = df_pid['total_reward']
        df_no_lag['profit'] = df_no_lag['total_reward']


    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'axes.labelsize': plot_cfg.label_fontsize,
        'xtick.labelsize': plot_cfg.tick_fontsize,
        'ytick.labelsize': plot_cfg.tick_fontsize,
        'legend.fontsize': plot_cfg.legend_fontsize
    })


    output_dir = REPO_ROOT / plot_cfg.save_dir
    output_dir.mkdir(parents=True, exist_ok=True)


    fig1, ax1 = plt.subplots(figsize=(plot_cfg.fig_width, plot_cfg.fig_height))

    ax1.plot(df_pid['Episode'], df_pid['profit'],
             label='PID-SACA', color=plot_cfg.pid_color,
             linewidth=plot_cfg.linewidth)
    ax1.plot(df_vanilla['Episode'], df_vanilla['profit'],
             label='Vanilla Lag.', color=plot_cfg.vanilla_color,
              linewidth=plot_cfg.linewidth, alpha=0.8)
    ax1.plot(df_no_lag['Episode'], df_no_lag['profit'],
             label='No Lag.', color=plot_cfg.no_lag_color,
              linewidth=plot_cfg.linewidth, alpha=0.8)

    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Profit')
    ax1.legend(frameon=True, loc='best', edgecolor='black', fancybox=False)
    ax1.grid(True, which='major', linestyle=plot_cfg.grid_linestyle, alpha=plot_cfg.grid_alpha)

    for spine in ax1.spines.values():
        spine.set_visible(True)

    fig1.tight_layout()
    save_path1 = output_dir / "profit_convergence.pdf"
    fig1.savefig(save_path1, format='pdf', bbox_inches='tight')
    print(f"Saved: {save_path1}")


    fig2, ax2 = plt.subplots(figsize=(plot_cfg.fig_width, plot_cfg.fig_height))

    ax2.plot(df_pid['Episode'], df_pid['lambda'],
             label='PID-SACA', color=plot_cfg.pid_color,
             linewidth=plot_cfg.linewidth)
    ax2.plot(df_vanilla['Episode'], df_vanilla['lambda'],
             label='Vanilla Lag.', color=plot_cfg.vanilla_color,
             linestyle='--', linewidth=plot_cfg.linewidth, alpha=0.8)
    ax2.plot(df_no_lag['Episode'], df_no_lag['lambda'],
             label='No Lag.', color=plot_cfg.no_lag_color,
             linestyle='--', linewidth=plot_cfg.linewidth, alpha=0.8)

    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Multiplier Value')
    ax2.legend(frameon=True, loc='lower right', edgecolor='black', fancybox=False)
    ax2.grid(True, which='major', linestyle=plot_cfg.grid_linestyle, alpha=plot_cfg.grid_alpha)

    for spine in ax2.spines.values():
        spine.set_visible(True)

    fig2.tight_layout()
    save_path2 = output_dir / "multiplier_dynamics.pdf"
    fig2.savefig(save_path2, format='pdf', bbox_inches='tight')
    print(f"Saved: {save_path2}")

    plt.show()


if __name__ == "__main__":





    plot_results()
