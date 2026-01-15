"""
Delta (Target ORR) 敏感性实验
测试不同 target_orr 值对模型性能的影响
"""
import os
import glob
import pandas as pd
import numpy as np
import pathlib
import PID_SACA as core

BASEFILE = pathlib.Path(__file__).parent.resolve()

# ==========================================
# 1. 实验配置类
# ==========================================
class DeltaTestConfig:
    """
    Delta 敏感性实验配置
    """
    def __init__(self):
        # Delta (target_orr) 取值数组
        # self.delta_values = [0.10, 0.25, 0.40, 0.55, 0.70]
        self.delta_values = [0.10, 0.25, 0.40, 0.55, 0.70, 0.80, 0.90, 1.00]
        
        # 训练参数
        self.episodes = 200           # 每个实验的训练轮次
        
        # 结果分析参数
        self.avg_last_n = 10          # 计算最后 N 步的平均值
        
        # 文件路径
        self.log_dir = BASEFILE / "log"                    # 训练日志保存目录
        self.figure_dir = BASEFILE / "figure" / "delta"    # 绘图数据目录
        self.log_pattern = "pid_saca_training_log_*.csv"   # 日志文件匹配模式


# ==========================================
# 2. 训练运行函数
# ==========================================
def run_delta_experiments(config: DeltaTestConfig):
    """
    循环运行不同 delta (target_orr) 值的训练实验
    """
    print("=" * 60)
    print("Starting Delta Sensitivity Experiments")
    print(f"Delta values to test: {config.delta_values}")
    print(f"Episodes per experiment: {config.episodes}")
    print("=" * 60)
    
    for i, delta in enumerate(config.delta_values):
        print(f"\n>>> [{i+1}/{len(config.delta_values)}] Running experiment with delta (target_orr) = {delta}")
        
        # 创建训练配置
        train_cfg = core.Config()
        train_cfg.target_orr = delta
        train_cfg.episodes = config.episodes
        train_cfg.max_episodes = config.episodes
        
        # 运行训练
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


# ==========================================
# 3. 结果绘图/分析函数
# ==========================================
def plot_delta_results(config: DeltaTestConfig):
    """
    读取 figure/delta 目录下最新的几个日志文件，
    计算每个实验的 Profit 和 ORR 平均值，输出结果表格
    """
    print("\n>>> Analyzing Delta Sensitivity Results...")
    
    # 1. 获取日志文件
    search_path = config.figure_dir / config.log_pattern
    files = glob.glob(str(search_path))
    
    num_deltas = len(config.delta_values)
    
    if len(files) < num_deltas:
        print(f"Warning: Expected {num_deltas} log files, but found {len(files)}")
        print(f"Search path: {search_path}")
        if len(files) == 0:
            print("No files found. Please run experiments first or copy log files to figure/delta/")
            return None
    
    # 按修改时间排序，取最新的 num_deltas 个文件
    files_sorted = sorted(files, key=os.path.getmtime)
    latest_files = files_sorted[-num_deltas:]
    
    print(f"Found {len(latest_files)} log files:")
    for f in latest_files:
        print(f"  - {os.path.basename(f)}")
    
    # 2. 读取数据并计算指标
    results = []
    
    for i, (delta, file_path) in enumerate(zip(config.delta_values, latest_files)):
        try:
            df = pd.read_csv(file_path)
            
            # 取最后 N 步计算平均值
            last_n = df.tail(config.avg_last_n)
            
            # 计算 Profit (revenue) 和 ORR 的平均值
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
    
    # 3. 创建结果表格
    df_results = pd.DataFrame(results)
    df_results = df_results.set_index('delta')
    
    # 转置表格：行=指标，列=delta值
    df_transposed = df_results.T
    
    print("\n" + "=" * 60)
    print(f"Delta Sensitivity Results (Average over last {config.avg_last_n} episodes)")
    print("=" * 60)
    print(df_transposed.to_string())
    print("=" * 60)
    
    # 4. 保存结果
    output_path = config.figure_dir / "delta_sensitivity_results.csv"
    config.figure_dir.mkdir(parents=True, exist_ok=True)
    # df_transposed.to_csv(output_path)
    print(f"\nResults saved to: {output_path}")
    
    return df_transposed


# ==========================================
# 4. 主函数
# ==========================================
if __name__ == "__main__":
    # 实例化配置
    config = DeltaTestConfig()
    
    # 运行实验 (生成的 log 文件在 log 目录下，需要手动复制到 figure/delta)
    # run_delta_experiments(config)
    
    # 分析结果 (需要先将 log 文件复制到 figure/delta 目录)
    plot_delta_results(config)
