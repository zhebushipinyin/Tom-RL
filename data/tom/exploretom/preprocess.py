import pandas as pd
import numpy as np

def create_small_dataset(n=5000):
    # 读取原始训练数据
    train_df = pd.read_parquet('data/tom/exploretom/train.parquet')
    
    # 随机打乱并抽取5000个样本
    sampled_df = train_df.sample(n=n, random_state=42)
    
    # 保存到新文件
    sampled_df.to_parquet('data/tom/exploretom/train_5k.parquet')
    
    print(f"raw dataset length: {len(train_df)}")
    print(f"samples dataset length: {len(sampled_df)}")

if __name__ == "__main__":
    create_small_dataset()
