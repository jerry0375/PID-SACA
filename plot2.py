import matplotlib.pyplot as plt
import pandas as pd
import glob
import os

# --- 设置学术风格的全局字体 ---
# 这会让图表看起来更专业 (类似 Times New Roman)
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
# ---------------------------

def plot_comparison():
    log_dir2 = "figure/compare"
    log_dir = "log"
    log_dir = log_dir2
    max_episodes = 150  # 统一截断量，只绘制前180条数据
    
    # ============ 配置平滑系数 ============
    # alpha 介于 0~1 之间。
    # 0.1 非常平滑 (适合噪声极大的数据)
    # 0.3 中等平滑 (推荐起始值)
    # 0.8 几乎不平滑
    SMOOTHING_ALPHA = 0.2
    # ====================================

    # 定义需要对比的模型及其对应的文件前缀和绘图样式
    # 格式: (文件前缀模式, 图例标签, 颜色, 线型)
    models_config = [
    # 核心方法：使用深红色，实线，强调突出
        ("pid_saca_training_log_*.csv", "PID-SACA", "#D62728", "-"),  
        
        # 强对比方法：使用深蓝色，实线
        ("grc_elg_training_log_*.csv", "GRC-ELG", "#1F77B4", "-"), 
        
        # 次要对比：使用深橙色，实线
        ("saca_training_log_*.csv", "SACA", "#FF7F0E", "-"),   
        
        # 其他对比：使用森林绿，实线
        ("jdrl_training_log_*.csv", "JDRL-POMO", "#2CA02C", "-"), 
        
        # 基础 Baseline：使用灰色，虚线或实线均可，稍微降低存在感
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
            
        # 获取最新的文件
        latest_file = sorted(files)[-1]
        print(f"Found for {label}: {latest_file}")
        
        try:
            df = pd.read_csv(latest_file)
            
            # 应用截断
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

    # 绘图指标
    metrics = ["total_reward", "revenue", "cost", "orr"]
    summary_metrics = ["total_reward", "revenue", "cost", "orr", "total_distance"]  # 汇总表格包含距离
    ylabels = ["Reward", "Profit", "Cost", "ORR"]
    titles = ["Average Reward", "Average Revenue", "Average Cost", "Average ORR"]
    
    # 计算并打印对比表格 (这里使用原始数据的最后几项平均值，无需平滑)
    print("\n" + "="*60)
    print(f"Performance Comparison (Average over last 40 episodes of the first {max_episodes})")
    print("Baseline for %: Worst performing model = 100%")
    print("="*60)

    summary_data = {}
    
    for metric in summary_metrics:
        # 1. 计算每个模型的平均值
        model_means = {}
        for item in data_to_plot:
            df = item["df"]
            if metric in df.columns:
                # 取最后40条数据计算平均值
                model_means[item["label"]] = df[metric].tail(40).mean()
            else:
                model_means[item["label"]] = float('nan')
        
        # 2. 确定基准值 (最差的模型)
        valid_values = [v for v in model_means.values() if not pd.isna(v)]
        if not valid_values:
            continue
            
        if metric in ["cost", "total_distance"]:
            # Cost 和 Distance 越小越好，所以最大值是最差的
            baseline_value = max(valid_values)
        else:
            # 其他指标越大越好，所以最小值是最差的
            baseline_value = min(valid_values)
            
        # 3. 格式化输出
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

    # 创建汇总 DataFrame
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
                # ================= 平滑处理核心代码 =================
                # 使用 Pandas 的 ewm (指数加权移动平均) 进行平滑
                # adjust=False 保证了早期数据的平滑效果更符合直觉
                smoothed_data = df[metric].ewm(alpha=SMOOTHING_ALPHA, adjust=False).mean()
                
                # 注意：这里绘制的是 smoothed_data，而不是原始的 df[metric]
                plt.plot(df["Episode"], smoothed_data, 
                         label=item["label"], 
                         color=item["color"], 
                         linestyle=item["linestyle"], 
                         linewidth=2.5) # 稍微增加线宽，看起来更饱满
                # =================================================

            else:
                print(f"Metric '{metric}' not found for {item['label']}")
        
        # plt.title(titles[i])
        plt.xlabel("Episode")
        plt.ylabel(ylabels[i])
        # 优化图例显示，保留边框
        plt.legend(frameon=True, loc='best', edgecolor='black', fancybox=False)
        # 优化网格线，使用虚线且颜色更淡
        plt.grid(True, which='major', linestyle='--', alpha=0.4)
        # 移除顶部和右侧的脊柱线，使图表更干净
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


    plt.tight_layout()
    
    # 保存合并图片
    output_filename = "comparison_result_smoothed.png"
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"comparison_result_smoothed_{timestamp}.pdf" # 建议保存为 PDF 矢量图
    output_path = os.path.join(log_dir, output_filename)
    # 保存为 PDF 可以保证在论文中无限放大不失真
    # plt.savefig(output_path, format='pdf', bbox_inches='tight')
    print(f"\nComparison plot saved to '{output_path}'")
    
    # 分别保存四个子图的 PDF
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
        # 保留四边边框
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
    
    # 显示图片
    plt.show()

if __name__ == "__main__":
    plot_comparison()