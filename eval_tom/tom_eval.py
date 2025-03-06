import argparse
import json
import os
import numpy as np
import random
import torch
import time
from vllm import LLM, SamplingParams
import pandas as pd
from typing import Tuple, Optional, List, Dict, Any
import re
from concurrent.futures import ThreadPoolExecutor

# 使用更安全的方式设置多进程启动方法
try:
    import multiprocessing
    # 只在尚未设置时设置启动方法
    if not multiprocessing.get_start_method(allow_none=True):
        multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass  # 如果已经设置，则忽略错误

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
        print(f"[Error] No valid answer tags found, {processed_str}")
        # !!!
        if '</think>' in processed_str:
            final_answer = processed_str.split('</think>')[-1]
            return final_answer, processed_str
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
    
    return normalized


def init_seed(seed=42):
    """Initialize random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def process_row(row, model_answer, data_source: str = "exploretom"):
    """Process a single row with the model's answer.
    
    Args:
        row: DataFrame row containing the data
        model_answer: The model's answer for this row
        
    Returns:
        Dictionary containing processed data
    """
    extra_info = {}
    if data_source == "exploretom":
        story, question = row['infilled_story'], row['question']
        prompt_msgs = row['prompt']
        prompt = prompt_msgs[0]['content']
        gt_answer = row['reward_model']['ground_truth']
    elif data_source == 'hi_tom':
        story, question = row['story'], row['question_old']
        prompt = format_prompt(story + '\n\n' + question)
        # gt_answer = ' '.join(row['answer'].split('_'))
        gt_answer = row['answer']
        extra_info['question_order'] = row['question_order']
        extra_info['deception'] = row['deception']
    elif data_source == 'tom_i':
        story, question = row['story'], row['question']
        prompt = format_prompt(story + '\n\n' + question)
        # gt_answer = ' '.join(row['answer'].split('_'))
        gt_answer = row['answer']
        extra_info['question_type'] = row['question_type']
    processed_model_answer, _ = extract_solution(model_answer)
    processed_model_answer = normalize_answer(processed_model_answer) if processed_model_answer else ""
    gt_answer = normalize_answer(gt_answer)
    
    return {
        'story': story,
        'question': question,
        'prompt': prompt,
        'gt_answer': gt_answer,
        'model_answer': model_answer,
        'processed_model_answer': processed_model_answer,
        'extra_info': extra_info
    }

def batch_process(df: pd.DataFrame, llm: LLM, sampling_params: SamplingParams, 
                  batch_size: int = 16, data_source: str = "exploretom") -> List[Dict[str, Any]]:
    """Process dataframe in batches for more efficient inference.
    
    Args:
        df: DataFrame containing the data
        llm: Language model instance
        sampling_params: Sampling parameters for generation
        batch_size: Number of examples to process in each batch
        
    Returns:
        List of dictionaries containing processed results
    """
    all_results = []
    total_examples = len(df)
    
    for i in range(0, total_examples, batch_size):
        batch_end = min(i + batch_size, total_examples)
        batch_df = df.iloc[i:batch_end]
        
        # Extract prompts for this batch
        if data_source == "exploretom":
            prompts = [row['prompt'][0]['content'] for _, row in batch_df.iterrows()]
        elif data_source == 'hi_tom':
            prompts = [format_prompt(row['story'].replace('\n', ' ')+ '\n\n' + row['question_old'])  for _, row in batch_df.iterrows()]
        elif data_source == 'tom_i':
            prompts = [format_prompt(row['story'].replace('\n', ' ')+ '\n\n' + row['question']) for _, row in batch_df.iterrows()]
        
        # Generate outputs for all prompts in the batch
        print(f"Processing batch {i//batch_size + 1}/{(total_examples + batch_size - 1)//batch_size} ({i}-{batch_end-1})")
        outputs = llm.generate(prompts, sampling_params)
        
        # Process each row with its corresponding output
        batch_results = []
        for j, (_, row) in enumerate(batch_df.iterrows()):
            model_answer = outputs[j].outputs[0].text
            result = process_row(row, model_answer, data_source)
            batch_results.append(result)
            
        all_results.extend(batch_results)
        
    return all_results

def format_prompt(quiz: str):
    template = f"""<|im_start|>system\nYou are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. Now the user asks you to solve a theory of mind reasoning problem. After thinking, when you finally reach a conclusion, clearly state your answer within <answer> </answer> tags.\n<|im_end|>\n<|im_start|>user\n{quiz}\n<|im_end|>\n<|im_start|>assistant\n<think>"""
    return template

def main(args):
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.ngpus,
        max_model_len=args.max_token,
        gpu_memory_utilization=0.85,
        enforce_eager=False
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_token,
    )

    # Load the parquet file
    if args.data_dir.endswith('.parquet'):
        df = pd.read_parquet(args.data_dir)
    else:
        df = pd.read_csv(args.data_dir)
    
    print(f"Processing {len(df)} examples using batch processing")
    start_time = time.time()
    
    # Process in batches for more efficient inference
    results = batch_process(df, llm, sampling_params, batch_size=args.batch_size, data_source=args.data_source)
    
    # Extract results
    stories = [result['story'] for result in results]
    questions = [result['question'] for result in results]
    prompts = [result['prompt'] for result in results]
    gt_answers = [result['gt_answer'] for result in results]
    model_answers = [result['model_answer'] for result in results]
    processed_model_answers = [result['processed_model_answer'] for result in results]
    extra_info = [result['extra_info'] for result in results]
    # Calculate accuracy
    correct = sum(1 for gt, model in zip(gt_answers, processed_model_answers) if gt == model)
    accuracy = correct / len(gt_answers) if gt_answers else 0
    
    print(f"Evaluation completed in {time.time() - start_time:.2f} seconds")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{len(gt_answers)})")
    
    # Save results
    os.makedirs(args.save_dir, exist_ok=True)
    model_name = os.path.basename(args.model)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    result_file = os.path.join(args.save_dir, f"results_{args.data_source}_{model_name}_{timestamp}.json")
    
    # Create a DataFrame for Excel export
    df_results = pd.DataFrame({
        "story": stories,
        "question": questions,
        "prompt": prompts,
        "model_answer": model_answers,
        "processed_model_answer": processed_model_answers,
        "ground_truth": gt_answers,
        "is_correct": [gt == processed for gt, processed in zip(gt_answers, processed_model_answers)],
        # Expand extra_info dictionary into separate columns
        **{f"extra_{key}": [info.get(key, None) for info in extra_info] 
           for key in (extra_info[0].keys() if extra_info and len(extra_info) > 0 else [])}
    })
    
    # Save as JSON for compatibility
    results_data = {
        "examples": [
            {
                "story": story,
                "question": question,
                "prompt": prompt,
                "model_answer": model_answer,
                "processed_model_answer": processed_model_answer,
                "ground_truth": gt,
                "is_correct": gt == processed_model_answer,
                "extra_info": extra_info
            }
            for story, question, prompt, gt, model_answer, processed_model_answer, extra_info in zip(
                stories, questions, prompts, gt_answers, model_answers, processed_model_answers, extra_info
            )
        ]
    }
    
    # Save to both Excel and JSON formats
    excel_file = result_file.replace('.json', '.xlsx')
    df_results.to_excel(excel_file, index=False)
    
    with open(result_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"Results saved to {result_file}")

if __name__ == "__main__":
    import os
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    
    parser = argparse.ArgumentParser(description="Evaluation script for KK dataset")
    parser.add_argument("--data_dir", '-d', type=str, 
                        default="./data/tom/exploretom/test.parquet", help="Data directory")
    parser.add_argument("--data_source", type=str, 
                        default="exploretom", 
                        choices=["exploretom", "hi_tom", "tom_i"], 
                        help="Data source")
    parser.add_argument("--save_dir", "-s", type=str, default="./results", help="Save directory")
    parser.add_argument("--model", "-m", type=str, 
                        default='./checkpoints/GRPO_tom/Qwen-7B-IM/actor/global_step_700', 
                        help="Model name or path")
    parser.add_argument("--max_token", type=int, default=2048, help="Maximum number of tokens")
    parser.add_argument("--ngpus", type=int, default=2, help="Number of GPUs")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p (nucleus) sampling")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for processing")
    args = parser.parse_args()

    init_seed()
    main(args)