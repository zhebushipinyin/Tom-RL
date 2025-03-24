import os
import re
import pandas as pd
import glob

# 指定结果文件所在的文件夹
results_dir = "."  # 或者使用绝对路径，如 "/path/to/ckpts_results/"

def extract_model_info(filename):
    """从文件名中提取model_id和step信息"""
    # 提取基本信息
    match = re.search(r'Qwen2\.5-(.+?)_', filename)
    if not match:
        return None
    
    model_raw = match.group(1)
    
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
    
    return model_id

def analyze_results():
    """分析所有结果文件并生成统计信息"""
    # 获取所有CSV文件
    csv_files = glob.glob(os.path.join(results_dir, "*.csv"))
    
    # 创建结果字典，使用(model_id, step)作为键
    results_dict = {}
    
    # 处理每个文件
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        model_id = extract_model_info(filename)
        
        if model_id is None:
            print(f"skip {filename}, because it is not a valid model id")
            continue
        
        # 跳过explore_tom_test_2662类型的文件
        if "explore_tom_test_2662" in filename:
            print(f"skip {filename}, because it is explore_tom_test_2662 type")
            continue
        
        # 读取CSV文件
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"error reading {filename}: {e}")
            continue
        
        # 获取或创建结果字典
        key = (model_id)
        if key not in results_dict:
            results_dict[key] = {
                "model_id": model_id,
                "hi_tom_600": None,
                "hi_tom": None,
                "tomi": None,
                "explore_tom": None,
                "explore_tom2": None  # 新增字段
            }
            
        # 处理Hi_ToM_cleaned类型的文件
        if "Hi_ToM_cleaned" in filename:
            if "rule_based_eval" in df.columns:
                results_dict[key]["hi_tom_600"] = df["rule_based_eval"].mean()
                
        # 处理ToM_test_HiExTi_hint类型的文件
        elif "ToM_test_HiExTi_hint" in filename:
            if "data_source" in df.columns and "rule_based_eval" in df.columns:
                # 检查是否是版本2的文件
                is_version2 = filename.endswith("_2.csv")
                
                # 按data_source分组计算准确率
                accuracy_by_source = df.groupby("data_source")["rule_based_eval"].mean()
                
                # 填充相应的结果
                for source in accuracy_by_source.index:
                    accuracy = accuracy_by_source[source]
                    
                    # 对于explore_tom，根据文件版本选择不同的目标字段
                    if source == "explore_tom":
                        if is_version2:
                            results_dict[key]["explore_tom2"] = accuracy
                        else:
                            results_dict[key]["explore_tom"] = accuracy
                    elif source in ["hi_tom", "tomi"]:
                        results_dict[key][source] = accuracy
        
    # 创建结果数据框并排序
    if results_dict:
        results_df = pd.DataFrame(list(results_dict.values()))
        results_df = results_df.sort_values(by=["model_id"])
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