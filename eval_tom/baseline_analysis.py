import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 文件路径
files = [
    "./basline/Qwen2.5-0.5B-Instruct_hi_tom_3000.csv",
    "./basline/Qwen2.5-1.5B-Instruct_hi_tom_3000.csv",
    "./basline/Qwen2.5-3B-Instruct_hi_tom_3000.csv",
    "./basline/Qwen2.5-7B-Instruct_hi_tom_3000.csv"
]

# 创建2x2的子图布局
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()

# 对每个文件进行分析
for i, file in enumerate(files):
    # 读取CSV文件
    df = pd.read_csv(file)
    
    # 按question_order分组并计算rule_based_eval的准确率
    accuracy_by_order = df.groupby('question_order')['rule_based_eval'].mean()
    
    # 绘制子图
    ax = axes[i]
    ax.bar(accuracy_by_order.index, accuracy_by_order.values)
    
    # 在每个柱状图上添加准确率数值
    for j, v in enumerate(accuracy_by_order.values):
        ax.text(j, v + 0.01, f'{v:.2f}', ha='center')
    
    # 设置子图标题和标签
    model_name = file.split('_')[0]
    ax.set_title(f'{model_name} accuracy analysis')
    ax.set_xlabel('Question Order')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.0)
    
    # 添加网格线以提高可读性
    ax.grid(True, linestyle='--', alpha=0.7)

# 调整布局
plt.tight_layout()

# 为整个图添加总标题
# to English
plt.suptitle('Accuracy Comparison of Qwen2.5 Models in Question Order', fontsize=16)
plt.subplots_adjust(top=0.92)

# 保存图像
output_dir = 'results'
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'question_order_accuracy.png'), dpi=300)



