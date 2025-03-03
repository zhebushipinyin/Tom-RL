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

def process_row(row, model_answer):
    """Process a single row with the model's answer.
    
    Args:
        row: DataFrame row containing the data
        model_answer: The model's answer for this row
        
    Returns:
        Dictionary containing processed data
    """
    infilled_story, question = row['infilled_story'], row['question']
    prompt_msgs = row['prompt']
    prompt = prompt_msgs[0]['content']
    gt_answer = row['reward_model']['ground_truth']

    processed_model_answer, _ = extract_solution(model_answer)
    processed_model_answer = normalize_answer(processed_model_answer) if processed_model_answer else ""
    gt_answer = normalize_answer(gt_answer)
    
    return {
        'infilled_story': infilled_story,
        'question': question,
        'prompt': prompt,
        'gt_answer': gt_answer,
        'model_answer': model_answer,
        'processed_model_answer': processed_model_answer
    }

def batch_process(df: pd.DataFrame, llm: LLM, sampling_params: SamplingParams, batch_size: int = 16) -> List[Dict[str, Any]]:
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
        prompts = [row['prompt'][0]['content'] for _, row in batch_df.iterrows()]
        
        # Generate outputs for all prompts in the batch
        print(f"Processing batch {i//batch_size + 1}/{(total_examples + batch_size - 1)//batch_size} ({i}-{batch_end-1})")
        outputs = llm.generate(prompts, sampling_params)
        
        # Process each row with its corresponding output
        batch_results = []
        for j, (_, row) in enumerate(batch_df.iterrows()):
            model_answer = outputs[j].outputs[0].text
            result = process_row(row, model_answer)
            batch_results.append(result)
            
        all_results.extend(batch_results)
        
    return all_results

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
    df = pd.read_parquet(args.data_dir)
    
    print(f"Processing {len(df)} examples using batch processing")
    start_time = time.time()
    
    # Process in batches for more efficient inference
    results = batch_process(df, llm, sampling_params, batch_size=args.batch_size)
    
    # Extract results
    infilled_stories = [result['infilled_story'] for result in results]
    questions = [result['question'] for result in results]
    prompts = [result['prompt'] for result in results]
    gt_answers = [result['gt_answer'] for result in results]
    model_answers = [result['model_answer'] for result in results]
    processed_model_answers = [result['processed_model_answer'] for result in results]
    
    # Calculate accuracy
    correct = sum(1 for gt, model in zip(gt_answers, processed_model_answers) if gt == model)
    accuracy = correct / len(gt_answers) if gt_answers else 0
    
    print(f"Evaluation completed in {time.time() - start_time:.2f} seconds")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{len(gt_answers)})")
    
    # Save results
    os.makedirs(args.save_dir, exist_ok=True)
    model_name = os.path.basename(args.model)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    result_file = os.path.join(args.save_dir, f"results_{model_name}_{timestamp}.json")
    
    # save to excel
    # Create a DataFrame for Excel export
    df_results = pd.DataFrame({
        "infilled_story": infilled_stories,
        "question": questions,
        "prompt": prompts,
        "model_answer": model_answers,
        "processed_model_answer": processed_model_answers,
        "ground_truth": gt_answers,
        "is_correct": [gt == processed for gt, processed in zip(gt_answers, processed_model_answers)]
    })
    
    # Save as JSON for compatibility
    results_data = {
        "examples": [
            {
                "infilled_story": story,
                "question": question,
                "prompt": prompt,
                "model_answer": model_answer,
                "processed_model_answer": processed_model_answer,
                "ground_truth": gt,
                "is_correct": gt == processed_model_answer
            }
            for story, question, prompt, gt, model_answer, processed_model_answer in zip(
                infilled_stories, questions, prompts, gt_answers, model_answers, processed_model_answers
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
    parser.add_argument("--data_dir", "-d", type=str, 
                        default="./data/tom/exploretom/test.parquet", help="Data directory")
    parser.add_argument("--save_dir", "-s", type=str, default="./results", help="Save directory")
    parser.add_argument("--model", "-m", type=str, 
                        default='./checkpoints/GRPO_tom/Qwen-7B-IM/actor/global_step_700', 
                        help="Model name or path")
    parser.add_argument("--max_token", type=int, default=2048, help="Maximum number of tokens")
    parser.add_argument("--ngpus", type=int, default=2, help="Number of GPUs")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p (nucleus) sampling")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for processing")
    args = parser.parse_args()

    init_seed()
    main(args)