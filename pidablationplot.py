import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
import numpy as np
import pathlib

BASEFILE = pathlib.Path(__file__).parent.resolve()

# ==========================================
# 1. 配置类 (方便调试参数)
# ==========================================
class Config:
    """PID消融实验分析配置"""
    def __init__(self):
        # 文件路径配置
        self.figure_dir = BASEFILE / "figure" / "pid"   # 日志文件目录
        self.pattern = "pid_saca_training_log_*.csv"    # 文件匹配模式
        
        # 分析参数配置
        self.avg_last_n = 100           # 统计最后 N 步的平均值
        self.max_episodes = 200         # 数据截断点 (可选)
        
        # 实验名称配置 (按文件更新时间从旧到新排序)
        self.experiment_names = [
            "Vanilla Lagrangian",       # 最旧的文件
            "PID Lagrangian",           # 中间的文件
            "No Lagrangian"             # 最新的文件
        ]
        
        # 统计指标配置
        self.metrics = ["revenue", "orr", "total_distance", "cost", "lambda"]
        
        # 输出配置
        self.output_filename = "pid_ablation_summary.csv"


# ==========================================
# 2. 辅助函数
# ==========================================
def get_sorted_files(config: Config):
    """
    获取目录下符合模式的文件，按修改时间排序（从旧到新）
    """
    search_path = config.figure_dir / config.pattern
    files = glob.glob(str(search_path))
    
    if not files:
        print(f"Error: No files found matching pattern: {search_path}")
        return []
    
    # 按修改时间排序（从旧到新）
    files_sorted = sorted(files, key=os.path.getmtime)
    return files_sorted


# ==========================================
# 3. 数据分析模块
# ==========================================
def analyze_results(config: Config):
    """
    读取日志文件，计算指标并输出表格。
    """
    print("\n" + "=" * 80)
    print(f"PID Ablation Study Analysis (Average over last {config.avg_last_n} episodes)")
    print("=" * 80)
    
    # 获取排序后的文件列表
    files = get_sorted_files(config)
    
    if len(files) < len(config.experiment_names):
        print(f"Warning: Expected {len(config.experiment_names)} files, but found {len(files)}")
    
    # 取最新的 N 个文件（N = 实验数量）
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
            
            # 数据截断（可选）
            if config.max_episodes and len(df) > config.max_episodes:
                df = df.iloc[:config.max_episodes]
            
            if len(df) == 0:
                print(f"Warning: Empty data for {exp_name}")
                continue
            
            # 计算最后 N 个 episode 的平均值
            last_n = df.tail(config.avg_last_n)
            exp_res = {}
            
            for m in config.metrics:
                if m in df.columns:
                    if m == "lambda":
                        # lambda 计算所有数据的方差
                        exp_res[m] = df[m].var()
                    else:
                        # 其他指标计算最后 N 步的平均值
                        exp_res[m] = last_n[m].mean()
                else:
                    exp_res[m] = float('nan')
                    print(f"  Warning: Metric '{m}' not found in {os.path.basename(file_path)}")
            
            results[exp_name] = exp_res
            print(f"\nLoaded: {exp_name} <- {os.path.basename(file_path)}")
            print(f"  Episodes: {len(df)}, Analyzing last {min(config.avg_last_n, len(df))} episodes")
            
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
    
    # 生成表格
    if not results:
        print("No results to display.")
        return None
    
    # 转置 DataFrame: 行=模型, 列=指标
    df_summary = pd.DataFrame(results).T
    
    # 确保列按配置顺序排列
    available_metrics = [m for m in config.metrics if m in df_summary.columns]
    df_summary = df_summary[available_metrics]
    
    # 重命名列（可读性更好）
    column_rename = {
        "revenue": "Profit",
        "orr": "ORR",
        "total_distance": "Dist.(km)",
        "cost": "Cost",
        "lambda": "Lambda Var."
    }
    df_summary = df_summary.rename(columns=column_rename)
    
    # 格式化输出
    print("\n" + "=" * 80)
    print("Summary Table:")
    print("=" * 80)
    pd.set_option('display.float_format', lambda x: '%.4f' % x)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(df_summary.to_string())
    print("=" * 80)
    
    # 保存结果
    output_path = config.figure_dir / config.output_filename
    # df_summary.to_csv(output_path)
    print(f"\nSummary table saved to: {output_path}")
    
    return df_summary


# ==========================================
# 主入口
# ==========================================
if __name__ == "__main__":
    # 实例化配置
    config = Config()
    
    # 分析结果
    analyze_results(config)

