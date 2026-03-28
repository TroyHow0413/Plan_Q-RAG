import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# 设置全局样式
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示
plt.rcParams['axes.unicode_minus'] = False 

def process_and_plot_all(file_path, window=5):
    # --- 1. 数据读取与清理 ---
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 过滤掉 Markdown 的分割线行 (如 | :---: |)
    clean_lines = [line for line in lines if ":---" not in line]
    raw_data = "".join(clean_lines)
    
    # 读取数据，处理竖线分隔符
    df = pd.read_csv(StringIO(raw_data), sep='|', skipinitialspace=True)
    df = df.dropna(axis=1, how='all') # 移除两端空列
    df.columns = [c.strip() for c in df.columns] # 清理列名空格

    # --- 2. 核心特征工程 ---
    # 提取 Step 数值 (从 "100/80000" 中提取 100)
    df['Step_Count'] = df['it/Total Steps'].apply(lambda x: int(str(x).split('/')[0]))
    
    # 处理时间：假设 Progress 列是累积时间 (格式 00:03:32)
    def time_to_hours(t_str):
        h, m, s = map(int, t_str.split(':'))
        return h + m/60.0 + s/3600.0
    
    df['Hours'] = df['Progress'].apply(time_to_hours)

    # --- 3. 开始独立绘图 ---
    
    # 图 1: Training Reward (随小时变化)
    draw_single_plot(df, 'Hours', 'Reward', '训练奖励趋势 (Train Reward)', 'Time (hours)', 'Reward', 'blue', 'reward_train.pdf', window)

    # 图 2: Eval Reward (随小时变化)
    draw_single_plot(df, 'Hours', 'Eval Reward', '评估奖励趋势 (Eval Reward)', 'Time (hours)', 'Return', 'green', 'reward_eval.pdf', window)

    # 图 3: QF Loss (对数坐标)
    plt.figure(figsize=(8, 6))
    plt.plot(df['Hours'], df['QF loss'], color='red', marker='o', markersize=3, alpha=0.7)
    plt.yscale('log')
    plt.title('Q-Function Loss (收敛分析)', fontsize=14)
    plt.xlabel('Time (hours)', fontsize=12)
    plt.ylabel('Loss (Log Scale)', fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.tight_layout()
    plt.savefig("qf_loss.pdf")
    plt.show()

    # 图 4: 训练效率 (Step vs Time)
    plt.figure(figsize=(8, 6))
    plt.plot(df['Hours'], df['Step_Count'], color='purple', linewidth=2)
    plt.title('训练吞吐量 (Throughput)', fontsize=14)
    plt.xlabel('Time (hours)', fontsize=12)
    plt.ylabel('Total Steps', fontsize=12)
    plt.tight_layout()
    plt.savefig("training_throughput.pdf")
    plt.show()

def draw_single_plot(df, x, y, title, xlabel, ylabel, color, filename, window):
    plt.figure(figsize=(8, 6))
    # 原始数据（浅色线）
    plt.plot(df[x], df[y], alpha=0.2, color=color)
    # 移动平均线（深色线）
    smooth_y = df[y].rolling(window=window, min_periods=1).mean()
    plt.plot(df[x], smooth_y, color=color, linewidth=2, label=f'SMA (n={window})')
    
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

# 使用：
process_and_plot_all('img_analysis\\100_4090_full\\log_100_4090_full copy.txt')