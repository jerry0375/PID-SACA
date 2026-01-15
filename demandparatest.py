"""
需求参数 (Bus Capacity) 敏感性实验
测试不同 bus_capacity 值对各算法性能的影响
"""
import os
import glob
import pandas as pd
import numpy as np
import pathlib

# 导入各算法模块
import PID_SACA
import GRC_ELG
import JDRL_POMO
import SACA
import SACA_baseline as SACA_MLP

BASEFILE = pathlib.Path(__file__).parent.resolve()

# ==========================================
# 1. 实验配置类
# ==========================================
class DemandParaTestConfig:
    """
    需求参数敏感性实验配置
    """
    def __init__(self):
        # Bus Capacity 取值数组 (改变供需关系)
        self.capacity_values = [20, 40, 60, 80]
        
        # 训练参数
        self.episodes = 150           # 每个实验的训练轮次
        
        # 结果分析参数
        self.avg_last_n = 40          # 计算最后 N 步的平均值
        
        # 文件路径
        self.log_dir = BASEFILE / "log"                     # 训练日志保存目录
        self.figure_dir = BASEFILE / "figure" / "demand"    # 绘图数据目录
        
        # 算法配置：(模块, 训练函数名, 日志文件模式, 显示名称)
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


# ==========================================
# 2. 训练运行函数
# ==========================================
def run_demand_experiments(config: DemandParaTestConfig):
    """
    循环运行不同 bus_capacity 值和不同算法的训练实验
    """
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
                # 获取模块和配置
                module = alg["module"]
                train_func = getattr(module, alg["train_func"])
                
                # 创建训练配置
                train_cfg = module.Config()
                train_cfg.bus_capacity = capacity
                
                # 设置训练轮次
                if hasattr(train_cfg, 'episodes'):
                    train_cfg.episodes = config.episodes
                if hasattr(train_cfg, 'max_episodes'):
                    train_cfg.max_episodes = config.episodes
                if hasattr(train_cfg, 'min_episodes'):
                    train_cfg.min_episodes = config.episodes
                
                # 运行训练
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


# ==========================================
# 3. 结果绘图/分析函数
# ==========================================
def plot_demand_results(config: DemandParaTestConfig):
    """
    读取 figure/demand 目录下的日志文件，
    计算每个算法在不同 capacity 下的 Profit, ORR, Cost, Dist.(km) 平均值，输出结果表格
    
    表格格式：
    - 行表头第一层：capacity 值 (20, 40, 60, 80)
    - 行表头第二层：指标 (Profit, ORR, Cost, Dist.(km))
    - 列表头：算法名称
    """
    print("\n>>> Analyzing Demand Parameter Sensitivity Results...")
    
    num_capacities = len(config.capacity_values)
    num_algorithms = len(config.algorithms)
    
    # 指标列表
    metrics = ['Profit', 'ORR', 'Cost', 'Dist.(km)']
    
    # 存储结果：{algorithm_name: {capacity: {metric: val}}}
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
        
        # 按修改时间排序，取最新的 num_capacities 个文件
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
    
    # 构建多级索引表格
    # 行：MultiIndex (capacity, metric)
    # 列：算法名称
    
    row_tuples = []
    for capacity in config.capacity_values:
        for metric in metrics:
            row_tuples.append((capacity, metric))
    
    row_index = pd.MultiIndex.from_tuples(row_tuples, names=['Capacity', 'Metric'])
    
    # 填充数据
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
    
    # 打印结果
    print("\n" + "=" * 80)
    print(f"Demand Parameter Sensitivity Results (Average over last {config.avg_last_n} episodes)")
    print("=" * 80)
    
    # 格式化输出
    pd.set_option('display.float_format', lambda x: f'{x:.4f}')
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(df_results.to_string())
    print("=" * 80)
    
    # 保存结果
    config.figure_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.figure_dir / "demand_sensitivity_results.csv"
    df_results.to_csv(output_path)
    print(f"\nResults saved to: {output_path}")
    
    return df_results


# ==========================================
# 4. 主函数
# ==========================================
if __name__ == "__main__":
    # 实例化配置
    config = DemandParaTestConfig()
    
    # 运行实验 (生成的 log 文件在 log 目录下，需要手动复制到 figure/demand)
    # run_demand_experiments(config)
    
    # 分析结果 (需要先将 log 文件复制到 figure/demand 目录)
    plot_demand_results(config)
