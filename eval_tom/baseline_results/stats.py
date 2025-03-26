import os
import re
import pandas as pd
import glob

# 指定结果文件所在的文件夹
results_dir = "."  # 或者使用绝对路径，如 "/path/to/ckpts_results/"

def normalize_answer(answer: str) -> str:
    """Normalizes the answer text for better comparison.
    Args:
        answer: Raw answer text
    Returns:
        Normalized answer text
    """
    if answer is None:
        return None
    # Convert to lowercase
    normalized = answer.lower()
    # Remove extra whitespace
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    # Remove punctuation that doesn't affect meaning
    normalized = re.sub(r'[.,;:!?]', '', normalized)
    normalized = re.sub(r'_', ' ', normalized)
    return normalized


def reward_func(response, answer):
    response_ = response
    # "answer": "answer text"
    try:
        response_ = response.split('answer": "')[1].split('"')[0]
    except Exception as e:
        return 0
    
    if response_:
        norm_response = normalize_answer(response_)
        norm_answer = normalize_answer(answer)

        # print(f'{norm_response} | {norm_answer}')
        # ans_pattern = r"\b(?:in|at|on|inside)?\s*(?:the\s*)?" + re.escape(norm_answer) + r"\b$"
        # match = re.match(ans_pattern, norm_response, re.DOTALL | re.MULTILINE)

        ans_pattern = r".*?" + re.escape(norm_answer) + r"\s*\b$"
        match = re.match(ans_pattern, norm_response)
        if match:
            return 1
        else:
            return 0
    return 0

def reevaluate_results(df, answer_column='answer'):
    """重新评估原来评估为错误的答案，提高召回率
    Args:
        df: 包含模型回答的DataFrame
        answer_column: 正确答案所在的列名
    Returns:
        添加了重新评估结果的DataFrame
    """
    # 复制原始DataFrame，添加新的评估列
    df = df.copy()
    df['expanded_rule_eval'] = df['rule_based_eval'].copy()
    
    # 只对原本为false的行进行重新评估
    false_indices = df[df['rule_based_eval'] == 0].index
    
    for idx in false_indices:
        # 重新使用reward_func进行评估
        expected_answer = df.loc[idx, answer_column] if answer_column in df.columns else df.loc[idx, 'expected_answer']
        model_answer = df.loc[idx, 'model_answer']
        
        # 重新评估
        new_eval = reward_func(model_answer, expected_answer)
        df.loc[idx, 'expanded_rule_eval'] = new_eval
    
    return df

def extract_model_info(filename):
    """从文件名中提取model_id和step信息"""
    # 提取基本信息
    match = re.search(r'Qwen2\.5-(.+?)_', filename)
    if not match:
        # 处理gpt-4o等其他格式的模型ID
        match = re.search(r'([\w\.-]+?(?:-\d{4}-\d{2}-\d{2})?)_', filename)
        if match:
            return match.group(1)
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
        
        # 跳过分析汇总文件
        if "analysis_summary" in filename:
            print(f"跳过分析汇总文件: {filename}")
            continue
            
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
            
            # 如果需要进行再评估，在内存中进行而不保存新文件
            expanded_eval_mean = None
            if 'model_answer' in df.columns and 'rule_based_eval' in df.columns:
                answer_column = 'answer' if 'answer' in df.columns else 'expected_answer'
                df_expanded = reevaluate_results(df, answer_column)
                expanded_eval_mean = df_expanded['expanded_rule_eval'].mean()
                # 打印改进信息
                rule_eval_mean = df['rule_based_eval'].mean()
                improvement = (expanded_eval_mean - rule_eval_mean) * 100
                print(f"{filename}: 原始准确率 {rule_eval_mean:.4f}, 扩展准确率 {expanded_eval_mean:.4f}, 提升 {improvement:.2f}%")
            
        except Exception as e:
            print(f"error reading {filename}: {e}")
            continue
        
        # 获取或创建结果字典
        key = (model_id)
        if key not in results_dict:
            results_dict[key] = {
                "model_id": model_id,
                "hi_tom_600": None,
                "hi_tom_600_expanded": None,
                "hi_tom": None,
                "hi_tom_expanded": None,
                "tomi": None,
                "tomi_expanded": None,
                "explore_tom": None,
                "explore_tom_expanded": None,
                "explore_tom2": None,
                "explore_tom2_expanded": None
            }
            
        # 处理Hi_ToM_cleaned类型的文件
        if "Hi_ToM_cleaned" in filename:
            if "rule_based_eval" in df.columns:
                results_dict[key]["hi_tom_600"] = df["rule_based_eval"].mean()
                # 添加扩展评估结果
                if expanded_eval_mean is not None:
                    results_dict[key]["hi_tom_600_expanded"] = expanded_eval_mean
                
        # 处理ToM_test_HiExTi_hint类型的文件
        elif "ToM_test_HiExTi_hint" in filename:
            if "data_source" in df.columns and "rule_based_eval" in df.columns:
                # 检查是否是版本2的文件
                is_version2 = filename.endswith("_2.csv") or "_2_" in filename
                
                # 按data_source分组计算准确率
                accuracy_by_source = df.groupby("data_source")["rule_based_eval"].mean()
                
                # 如果存在扩展评估结果，也计算
                expanded_accuracy_by_source = None
                if expanded_eval_mean is not None:
                    # 在内存中计算扩展评估的分组准确率
                    expanded_accuracy_by_source = df_expanded.groupby("data_source")["expanded_rule_eval"].mean()
                
                # 填充相应的结果
                for source in accuracy_by_source.index:
                    accuracy = accuracy_by_source[source]
                    
                    # 对于版本2的文件，只使用explore_tom数据，忽略hi_tom和tomi数据
                    if is_version2:
                        if source == "explore_tom":
                            results_dict[key]["explore_tom2"] = accuracy
                            if expanded_accuracy_by_source is not None:
                                results_dict[key]["explore_tom2_expanded"] = expanded_accuracy_by_source[source]
                    else:
                        # 非版本2文件的处理逻辑不变
                        if source == "explore_tom":
                            results_dict[key]["explore_tom"] = accuracy
                            if expanded_accuracy_by_source is not None:
                                results_dict[key]["explore_tom_expanded"] = expanded_accuracy_by_source[source]
                        elif source in ["hi_tom", "tomi"]:
                            results_dict[key][source] = accuracy
                            if expanded_accuracy_by_source is not None:
                                results_dict[key][f"{source}_expanded"] = expanded_accuracy_by_source[source]
        
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
    print("开始分析结果文件夹下的文件...(仅在内存中进行扩展评估，不创建新文件)")
    results_df = analyze_results()
    
    if not results_df.empty:
        # 显示结果
        print("\n分析结果:")
        print(results_df.to_string(index=False, na_rep="N/A"))
        
        # 保存结果到CSV - 确保不会有空行
        output_file = os.path.join(results_dir, "analysis_summary_expanded.csv")
        results_df.to_csv(output_file, index=False, na_rep="N/A")
        print(f"\n结果已保存到: {output_file}")
        
        # 打印改进信息
        print("\n各指标平均提升:")
        for col in results_df.columns:
            if col.endswith('_expanded') and not col.startswith('model_id'):
                base_col = col.replace('_expanded', '')
                if base_col in results_df.columns:
                    # 计算平均改进
                    orig_mean = results_df[base_col].mean(skipna=True)
                    expanded_mean = results_df[col].mean(skipna=True)
                    if not pd.isna(orig_mean) and not pd.isna(expanded_mean):
                        improvement = (expanded_mean - orig_mean) * 100
                        print(f"{base_col} -> {col}: 平均提升 {improvement:.2f}%")

if __name__ == "__main__":
    main()