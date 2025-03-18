import argparse
import os
import re
import pandas as pd
from vllm import LLM, SamplingParams
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from typing import Tuple, Optional


def extract_solution(solution_str: str) -> Tuple[Optional[str], str]:
    """Extracts the final answer from the model's response string.
    
    Args:
        solution_str: Raw response string from the language model
        
    Returns:
        Tuple containing (extracted_answer, processed_string)
    """
    # Split response to isolate assistant output
    # if "Assistant:" in solution_str:
    #     processed_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     print("[Error] Failed to locate model response header")
    #     return None, solution_str

    processed_str = solution_str
    # Extract final answer using XML-style tags
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))
    
    if not matches:
        print("[Error] No valid answer tags found")
        return None, processed_str
        
    final_answer = matches[-1].group(1).strip()
    return final_answer, processed_str

def normalize_answer(answer: str) -> str:
    """Normalizes the answer text for better comparison.
    
    Args:
        answer: Raw answer text
        
    Returns:
        Normalized answer text
    """
    # Convert to lowercase
    normalized = answer.lower()
    
    # Remove extra whitespace
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    # Remove punctuation that doesn't affect meaning
    normalized = re.sub(r'[.,;:!?]', '', normalized)

    # _ => ' '
    normalized = re.sub(r'_', ' ', normalized)
    return normalized

def check_answer_correctness(predicted_answer: str, ground_truth: str) -> Tuple[bool, float]:

    # ans_pattern = r".*?" + re.escape(norm_answer) + r"\s*\b$"

    """Checks if the predicted answer matches the ground truth.
    
    Args:
        predicted_answer: The answer extracted from model's response
        ground_truth: The ground truth answer
        
    Returns:
        Tuple containing (is_correct, score)
    """
    print("\n[Answer Validation]")
    print(f"  Ground truth: '{ground_truth}'")
    print(f"  Predicted: '{predicted_answer}'")
    
    # Normalize both answers for better comparison
    if not predicted_answer:
        print("  Predicted answer is empty")
        return False, 0
    norm_pred = normalize_answer(predicted_answer)
    
    norm_truth = normalize_answer(ground_truth)
    
    print(f"  Normalized ground truth: '{norm_truth}'")
    print(f"  Normalized prediction: '{norm_pred}'")
    
    # Check exact match after normalization
    if norm_pred == norm_truth:
        print("  Answer validation: EXACT MATCH with strict format")
        return True, 1
    
    ans_pattern = r".*?" + re.escape(norm_truth) + r"\s*\b$"
    if re.match(ans_pattern, norm_pred):
        print("  Answer validation: EXACT MATCH with loose format")
        return True, 1
    
    print("  Answer validation: MISMATCH")
    return False, 0

def make_prompt(story, question) -> str:
    quiz = story + "\n\n" + question
    prefix = f"""<|im_start|>system\nYou are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. Now the user asks you to solve a theory of mind reasoning problem. After thinking, when you finally reach a conclusion, clearly state your answer within <answer> </answer> tags.\n<|im_end|>\n<|im_start|>user\n{quiz}\n<|im_end|>\n<|im_start|>assistant\n<think>"""
    return prefix


def eval_model(model_path, data_path, output_dir, tp):
    llm = LLM(model=model_path, tokenizer=model_path, max_model_len=4096, tensor_parallel_size=tp)
    sampling_params = SamplingParams(
        max_tokens=4096,
        temperature=0.6,
        top_k=-1,
        top_p=0.95,
    )

    if data_path.endswith('.parquet'):
        dataset = load_dataset('parquet', data_files=data_path)['train']
    elif data_path.endswith('.csv'):
        dataset = load_dataset('csv', data_files=data_path)['train']
    elif data_path.endswith('.xlsx'):
        dataset = pd.read_excel(data_path)
        dataset = Dataset.from_pandas(dataset)
    else:
        raise ValueError(f"Unsupported file type: {data_path}")

    eval_prompts = []
    for example in dataset:
        if 'prompt' in example:
            prompt = example['prompt'][0]['content']
        elif 'story' in example and 'question' in example:
            prompt = make_prompt(example['story'], example['question'])
        elif 'story_structure' in example and 'question' in example:
            prompt = make_prompt(example['story_structure'], example['question'])
        else:
            raise ValueError(f"Invalid example: {example}")
        eval_prompts.append(prompt)
    model_results = llm.generate(eval_prompts, sampling_params, use_tqdm=True)

    results = []
    correct_count = 0
    for example, result in zip(dataset, model_results):
        model_answer = result.outputs[0].text
        example['model_answer'] = model_answer

        if 'answer' in example:
            gt_answer = example['answer']
        else:
            gt_answer = example['expected_answer']
    
        final_answer, processed_str = extract_solution(model_answer)
        example['final_answer'] = final_answer
        is_correct, score = check_answer_correctness(final_answer, gt_answer)
        example['is_correct'] = is_correct
        if 'extra_info' in example:
           # 展开 extra_info 
           for key, value in example['extra_info'].items():
               example[key] = value
           del example['extra_info']
        if is_correct:
            correct_count += 1
        results.append(example)

    print(f"{model_path} Accuracy: {correct_count}/{len(results)} = {correct_count / len(results)}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # global_step_500: Qwen2.5-7B-Instruct-1M-3e-7-True
    parser.add_argument("--model_path", type=str, default='./global_step_500/')
    parser.add_argument("--data_path", type=str, default='./data/cleaned_tom/raw/explore_tom.xlsx')
    parser.add_argument("--output_dir", type=str, default='./eval_tom/results/')
    parser.add_argument('--tp', type=int, default=2)
    args = parser.parse_args()
    eval_model(args.model_path, args.data_path, args.output_dir, args.tp)