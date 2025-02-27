import pandas as pd
import os
import shutil

# 定义目录路径
base_dir = 'data/kk/instruct'
difficulty_dirs = ['3ppl', '4ppl', '5ppl', '6ppl', '7ppl']
merge_dir = os.path.join(base_dir, 'merge')

# 创建 merge 目录（如果不存在）
if not os.path.exists(merge_dir):
    os.makedirs(merge_dir)
    print(f"Created directory: {merge_dir}")

# 初始化空的 DataFrame 用于合并
merged_train_df = pd.DataFrame()
merged_test_df = pd.DataFrame()

# 遍历每个难度配置目录
for difficulty in difficulty_dirs:
    dir_path = os.path.join(base_dir, difficulty)
    train_path = os.path.join(dir_path, 'train.parquet')
    test_path = os.path.join(dir_path, 'test.parquet')
    
    # 检查文件是否存在
    if os.path.exists(train_path) and os.path.exists(test_path):
        # 加载 parquet 文件
        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path)
        
        # 添加难度标识列
        train_df['difficulty'] = difficulty
        test_df['difficulty'] = difficulty
        
        # 合并数据
        merged_train_df = pd.concat([merged_train_df, train_df], ignore_index=True)
        merged_test_df = pd.concat([merged_test_df, test_df], ignore_index=True)
        
        print(f"Added {len(train_df)} train samples and {len(test_df)} test samples from {difficulty}")

# 保存合并后的数据集
merged_train_path = os.path.join(merge_dir, 'train.parquet')
merged_test_path = os.path.join(merge_dir, 'test.parquet')

merged_train_df.to_parquet(merged_train_path)
merged_test_df.to_parquet(merged_test_path)

# 打印合并结果
print(f"\nMerge complete!")
print(f"Total train samples: {len(merged_train_df)}")
print(f"Total test samples: {len(merged_test_df)}")
print(f"Merged train dataset saved to: {merged_train_path}")
print(f"Merged test dataset saved to: {merged_test_path}")

# # 复制数据分析文件到合并目录
# analysis_path = os.path.join(base_dir, 'dataset_analysis.md')
# if os.path.exists(analysis_path):
#     shutil.copy(analysis_path, os.path.join(merge_dir, 'dataset_analysis.md'))
#     print(f"Copied dataset analysis to merge directory")