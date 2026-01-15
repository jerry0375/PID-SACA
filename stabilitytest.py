import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import PID_SACA as core  

# ==========================================
# 1. 实验配置类 (Experiment Configuration)
# ==========================================
class ExpConfig:
    """
    实验控制台：在这里修改参数以进行 A/B 测试
    """
    def __init__(self):
        # 通用参数
        self.episodes = 200         # 训练轮次 (建议 150-200 以观察收敛)
        self.target_orr = 0.5       # 目标 ORR 约束
        self.seed = 42              # 固定随机种子，保证公平对比
        
        # --- Group A: Vanilla Lagrangian (Baseline) ---
        # 仅使用积分项 (I)，模拟标准 Lagrangian 方法
        # 特点：容易震荡
        self.vanilla_kp = 0.0
        self.vanilla_ki = 0.5      # 建议设置较小，否则震荡过于剧烈无法收敛
        self.vanilla_kd = 0.0
        
        # --- Group B: PID Lagrangian (Ours) ---
        # 使用完整的 PID 控制
        # 特点：快速收敛，稳定
        self.pid_kp = 5           # P项提供快速响应
        self.pid_ki = 1          # I项消除误差 (与 Vanilla 保持一致或略大)
        self.pid_kd = 5           # D项提供阻尼，抑制震荡


# ==========================================
# 1.5 绘图配置类 (Plot Configuration)
# ==========================================
class PlotConfig:
    """
    绘图参数配置：在这里调整图表样式
    """
    def __init__(self):
        # 图片尺寸 (16:9 比例)
        self.fig_width = 7
        self.fig_height = 6
        
        # 字体大小
        self.label_fontsize = 35      # 坐标轴标签字号
        self.tick_fontsize = 30       # 刻度字号
        self.legend_fontsize = 25     # 图例字号

        # 线条样式
        self.linewidth = 4          # 线条粗细
        self.grid_alpha = 0.4         # 网格透明度
        self.grid_linestyle = '--'    # 网格线型
        
        # 颜色配置
        self.pid_color = '#D62728'       # PID 曲线颜色 (深红)
        self.vanilla_color = '#1F77B4'   # Vanilla 曲线颜色 (深蓝)
        self.no_lag_color = '#2CA02C'    # No Lagrangian 曲线颜色 (森林绿)
        
        # 保存路径
        self.save_dir = "figure/pid"

# ==========================================
# 2. 训练逻辑 (Training Logic)
# ==========================================
def run_training(exp_config):
    """
    执行两组训练：Vanilla 和 PID
    """
    print("="*50)
    print("Starting Comparative Experiment: PID vs Vanilla")
    print("="*50)

    # --- 1. 运行 Group A: Vanilla Lagrangian ---
    print("\n>>> Running Group A: Vanilla Lagrangian...")
    
    # 获取原始配置并覆盖参数
    cfg_vanilla = core.Config()
    cfg_vanilla.episodes = exp_config.episodes
    cfg_vanilla.max_episodes = exp_config.episodes # 强制对齐
    cfg_vanilla.target_orr = exp_config.target_orr
    
    # 关键修改：设置为标准 Lagrangian (无 P, 无 D)
    cfg_vanilla.pid_kp = exp_config.vanilla_kp
    cfg_vanilla.pid_ki = exp_config.vanilla_ki
    cfg_vanilla.pid_kd = exp_config.vanilla_kd
    cfg_vanilla.pid_update_freq = 1
    
    # 调用核心训练函数 (传入 run_name 以便生成带标识的日志文件)
    # 注意：train_saca 内部会设置随机种子，确保环境初始化一致
    core.train_saca(cfg_vanilla)


    # --- 2. 运行 Group B: PID Lagrangian ---
    print("\n>>> Running Group B: PID Lagrangian (Ours)...")
    
    cfg_pid = core.Config()
    cfg_pid.episodes = exp_config.episodes
    cfg_pid.max_episodes = exp_config.episodes
    cfg_pid.target_orr = exp_config.target_orr
    
    # 关键修改：设置为 PID 参数
    cfg_pid.pid_kp = exp_config.pid_kp
    cfg_pid.pid_ki = exp_config.pid_ki
    cfg_pid.pid_kd = exp_config.pid_kd
    cfg_pid.pid_update_freq = 1
    
    core.train_saca(cfg_pid)

    print("\nAll training sessions completed.")

# ==========================================
# 3. 绘图与分析逻辑 (Plotting Logic)
# ==========================================
def get_latest_three_log_files():
    """辅助函数：获取 log 目录下最新的三个 pid_saca 训练日志文件"""
    # log_dir = pathlib.Path(__file__).parent.resolve() / "log"
    log_dir = pathlib.Path(__file__).parent.resolve() / "figure/pid"
    search_path = log_dir / "pid_saca_training_log_*.csv"
    files = glob.glob(str(search_path))
    if len(files) < 3:
        raise FileNotFoundError(f"Need at least 3 log files, but found {len(files)}")
    # 按修改时间排序
    files_sorted = sorted(files, key=os.path.getmtime)
    # 返回最新的三个文件：
    # [-3] 次次新 = Vanilla Lagrangian
    # [-2] 次新 = PID Lagrangian
    # [-1] 最新 = No Lagrangian
    return files_sorted[-3], files_sorted[-2], files_sorted[-1]

def plot_results(plot_cfg=None):
    """
    读取最新的日志文件并绘制对比图
    按照 run_training 的运行顺序：
    - 次次新：Vanilla Lagrangian
    - 次新：PID Lagrangian
    - 最新：No Lagrangian
    """
    if plot_cfg is None:
        plot_cfg = PlotConfig()
    
    print("\n>>> Generating Comparison Plots...")
    
    # 1. 读取数据
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

    # 3. 设置学术风格的全局字体
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'axes.labelsize': plot_cfg.label_fontsize,
        'xtick.labelsize': plot_cfg.tick_fontsize,
        'ytick.labelsize': plot_cfg.tick_fontsize,
        'legend.fontsize': plot_cfg.legend_fontsize
    })
    
    # 4. 创建保存目录
    output_dir = pathlib.Path(__file__).parent.resolve() / plot_cfg.save_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 5. 绘制并保存第一个子图: Profit Convergence
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
    # 保留四边边框
    for spine in ax1.spines.values():
        spine.set_visible(True)
    
    fig1.tight_layout()
    save_path1 = output_dir / "profit_convergence.pdf"
    fig1.savefig(save_path1, format='pdf', bbox_inches='tight')
    print(f"Saved: {save_path1}")
    
    # 6. 绘制并保存第二个子图: Multiplier Dynamics
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
    # 保留四边边框
    for spine in ax2.spines.values():
        spine.set_visible(True)
    
    fig2.tight_layout()
    save_path2 = output_dir / "multiplier_dynamics.pdf"
    fig2.savefig(save_path2, format='pdf', bbox_inches='tight')
    print(f"Saved: {save_path2}")
    
    plt.show()

# ==========================================
# 4. 主函数 (Main Execution)
# ==========================================
if __name__ == "__main__":
    # 1. 实例化配置
    # exp_config = ExpConfig()
    
    # # 2. 运行训练 (如果已经跑过数据，可以注释掉这行，直接跑绘图)
    # run_training(exp_config)
    
    # 3. 运行绘图
    plot_results()