import os
import re
import pandas as pd
import glob

# 指定结果文件所在的文件夹
results_dir = "."  # 或者使用绝对路径，如 "/path/to/ckpts_results/"

def extract_model_info(filename):
    """从文件名中提取model_id和step信息"""
    # 提取基本信息
    match = re.search(r'ckpt_Qwen2\.5-(.+?)_(\d+)_', filename)
    if not match:
        return None, None
    
    model_raw = match.group(1)
    step = int(match.group(2))
    
    # 处理特殊情况
    if "7B-Instruct-1M" in model_raw:
        model_id = "7B-1M"
    else:
        # 提取如0.5B这样的ID
        model_match = re.search(r'([\d\.]+B)', model_raw)
        if model_match:
            model_id = model_match.group(1)
        else:
            model_id = model_raw
    
    return model_id, step

def analyze_results():
    """分析所有结果文件并生成统计信息"""
    # 获取所有CSV文件
    csv_files = glob.glob(os.path.join(results_dir, "*.csv"))
    
    # 创建结果字典，使用(model_id, step)作为键
    results_dict = {}
    
    # 处理每个文件
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        model_id, step = extract_model_info(filename)
        
        if model_id is None or step is None:
            print(f"无法从{filename}中提取model_id和step信息，跳过")
            continue
        
        # 跳过explore_tom_test_2662类型的文件
        if "explore_tom_test_2662" in filename:
            print(f"跳过{filename}，因为它是explore_tom_test_2662类型")
            continue
        
        # 读取CSV文件
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"读取{filename}时出错: {e}")
            continue
        
        # 获取或创建结果字典
        key = (model_id, step)
        if key not in results_dict:
            results_dict[key] = {
                "model_id": model_id,
                "step": step,
                "hi_tom_600": None,
                "hi_tom": None,
                "tomi": None,
                "explore_tom": None
            }
            
        # 处理Hi_ToM_cleaned类型的文件
        if "Hi_ToM_cleaned" in filename:
            if "is_correct" in df.columns:
                results_dict[key]["hi_tom_600"] = df["is_correct"].mean()
                
        # 处理ToM_test_HiExTi_hint类型的文件
        elif "ToM_test_HiExTi_hint" in filename:
            if "data_source" in df.columns and "is_correct" in df.columns:
                # 按data_source分组计算准确率
                accuracy_by_source = df.groupby("data_source")["is_correct"].mean()
                
                # 填充相应的结果
                for source in ["hi_tom", "tomi", "explore_tom"]:
                    if source in accuracy_by_source:
                        results_dict[key][source] = accuracy_by_source[source]
    
    # 创建结果数据框并排序
    if results_dict:
        results_df = pd.DataFrame(list(results_dict.values()))
        results_df = results_df.sort_values(by=["model_id", "step"])
        return results_df
    else:
        print("没有找到可分析的结果文件")
        return pd.DataFrame()

def main():
    """主函数"""
    print("开始分析ckpts_results文件夹下的结果...")
    results_df = analyze_results()
    
    if not results_df.empty:
        # 显示结果
        print("\n分析结果:")
        print(results_df.to_string(index=False, na_rep="N/A"))
        
        # 保存结果到CSV
        output_file = os.path.join(results_dir, "analysis_summary.csv")
        results_df.to_csv(output_file, index=False)
        print(f"\n结果已保存到: {output_file}")

if __name__ == "__main__":
    main()