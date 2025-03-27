import argparse
import os
import re
import pandas as pd
import asyncio
import time
import json
import random
import numpy as np
from tqdm.asyncio import tqdm_asyncio
from datasets import load_dataset
from openai import AzureOpenAI, AsyncAzureOpenAI, AsyncOpenAI
from openai import RateLimitError
from dotenv import load_dotenv, find_dotenv

# SYSTEM_PROMPT = """Read the following story and answer the question. Think step-by-step. Provide the answer first,and then explain it. Answer in the following JSON format:
# {
# "answer": "answer text",
# "explain": "step by step thinking"
# }
# """

# SYSTEM_PROMPT = """Read the following story and answer the question. Think step-by-step. Provide the thinking first, and then the answer. \nNote: You should assume the following.\n(1) An agent witnesses everything and every movement before exiting a room.\n(2) An agent A can infer another agent B's mental state only if A and B have been in the same room, or have private or public interactions.\nAnswer in the following JSON format:
# {
# "thinking": "step by step thinking",
# "answer": "answer text"
# }
# """

SYSTEM_PROMPT = """Read the following story and answer the question. Think step-by-step. Provide the thinking first, and then the answer. Answer in the following JSON format:
{
"thinking": "step by step thinking",
"answer": "answer text"
}
"""

XML_COT_FORMAT = """Story: {}\n Question:{}"""

# 自定义JSON编码器，处理NumPy和pandas类型
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

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

# Add rate limiting and retry functionality for API calls
async def call_openai_async(client, model_id, prompt, max_retries=5, initial_delay=1):
    retry_count = 0
    while True:
        try:
            response = await client.chat.completions.create(
                model=model_id,
                messages=prompt,
                temperature=0.0,
                max_tokens=2048
            )
            return response.choices[0].message.content
        except RateLimitError as e:
            # Extract retry-after time if available from the error message
            retry_after = 5  # Default retry delay
            if hasattr(e, 'response'):
                message = str(e)
                # Try to extract the retry time from the error message
                match = re.search(r'retry after (\d+) seconds', message)
                if match:
                    retry_after = int(match.group(1))
            
            retry_count += 1
            if retry_count > max_retries:
                raise Exception(f"Maximum retries ({max_retries}) exceeded")
            
            # Add jitter to avoid all clients retrying at the same time
            jitter = random.uniform(0.5, 1.5)
            delay = initial_delay * (2 ** (retry_count - 1)) * jitter
            # Use the longer of calculated delay or API-suggested retry time
            delay = max(delay, retry_after)
            print(f"Rate limit hit. Retrying in {delay:.2f} seconds... (Attempt {retry_count}/{max_retries})")
            await asyncio.sleep(delay)
        except Exception as e:
            retry_count += 1
            if retry_count > max_retries:
                raise e
            delay = initial_delay * (2 ** (retry_count - 1))
            print(f"Error: {e}. Retrying in {delay:.2f} seconds... (Attempt {retry_count}/{max_retries})")
            await asyncio.sleep(delay)

def call_openai(client, model_id, prompt):
    response = client.chat.completions.create(
        model=model_id,
        messages=prompt,
        temperature=0.0,
        max_tokens=2048
    )
    return response.choices[0].message.content

async def eval_model_async(client, model_id, data_path, output_path, batch_size=10, checkpoint_interval=5):
    # 直接加载成pandas DataFrame
    print(f"Loading dataset from {data_path}")
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    elif data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    else:
        # 对于HuggingFace数据集，转换为pandas
        ds = load_dataset(data_path)['train']
        df = pd.DataFrame({col: ds[col] for col in ds.column_names})
    
    # FIXME, 只评估explore_tom
    df = df[df['data_source'] == 'explore_tom']

    # Create checkpoint dir if it doesn't exist
    checkpoint_dir = os.path.join(output_path, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Check for checkpoint
    data_id = data_path.split("/")[-1].split(".")[0]
    checkpoint_file = os.path.join(checkpoint_dir, f'{model_id}_{data_id}_checkpoint.json')
    start_batch = 0
    all_results = []
    total_rule_correct = 0
    
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                start_batch = checkpoint_data.get('next_batch', 0)
                total_rule_correct = checkpoint_data.get('total_correct', 0)
                # Load previous results if they exist
                prev_results_file = checkpoint_data.get('partial_results')
                if prev_results_file and os.path.exists(prev_results_file):
                    all_results = [pd.read_csv(prev_results_file)]
                print(f"Resuming from batch {start_batch} with {total_rule_correct} correct answers so far")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting from beginning.")
            start_batch = 0
            total_rule_correct = 0
            all_results = []
    
    print(f"Processing {len(df)} examples in batches of {batch_size}")
    
    total_batches = (len(df) + batch_size - 1) // batch_size
    
    # 批量处理
    for batch_idx in range(start_batch, total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(df))
        batch_df = df.iloc[start_idx:end_idx].copy()
        
        print(f"Processing batch {batch_idx + 1}/{total_batches}")
        
        # 为每个样本创建prompt
        prompts = []
        for _, row in batch_df.iterrows():
            if 'story' in batch_df.columns:
                # hi_tom
                story = row['story']
                if 'question_old' in batch_df.columns:
                    question = row['question_old']
                else:
                    question = row['question']
            else:
                # explore_tom
                story = row['story_structure']
                question = row['question']
            
            # FIXME, 只评估explore_tom 
            if 'data_source' in row and row['data_source'] == 'explore_tom':
                story = row['extra_info']['infilled_story']
                question = row['question']

            cot_prompt = XML_COT_FORMAT.format(story, question)
            prompt = [{'role': 'system', 'content': SYSTEM_PROMPT}, 
                      {'role': 'user', 'content': cot_prompt}]
            prompts.append(prompt)
        
        # 并行处理所有样本, but with rate limiting
        # Create tasks for semaphore to limit concurrency rate
        tasks = []
        for prompt in prompts:
            tasks.append(call_openai_async(client, model_id, prompt))
        
        # Process the tasks with a progress bar
        model_answers = await tqdm_asyncio.gather(*tasks)
        
        # 添加模型回答到DataFrame
        batch_df['model_answer'] = model_answers
        
        # 计算正确率
        batch_df['rule_based_eval'] = batch_df.apply(
            lambda row: reward_func(
                row['model_answer'], 
                row['answer'] if 'answer' in batch_df.columns else row['expected_answer']
            ),
            axis=1
        )
        
        batch_correct = batch_df['rule_based_eval'].sum()
        total_rule_correct += batch_correct
        all_results.append(batch_df)
        
        # Print progress
        current_count = sum(len(df) for df in all_results)
        print(f"Batch accuracy: {batch_correct}/{len(batch_df)} = {batch_correct/len(batch_df):.4f}")
        print(f"Overall accuracy so far: {total_rule_correct}/{current_count} = {total_rule_correct/current_count:.4f}")
        
        # Save checkpoint at regular intervals
        if (batch_idx + 1) % checkpoint_interval == 0 or batch_idx == total_batches - 1:
            # Save partial results
            partial_results = pd.concat(all_results, ignore_index=True)
            partial_file = os.path.join(checkpoint_dir, f'{model_id}_{data_id}_partial.csv')
            partial_results.to_csv(partial_file, index=False, encoding='utf-8-sig')
            
            # Save checkpoint info
            checkpoint_data = {
                'next_batch': batch_idx + 1,
                'total_correct': total_rule_correct,
                'partial_results': partial_file,
                'timestamp': time.time()
            }
            try:
                # Print data types for debugging if needed
                if batch_idx < 2:  # 只在前几个批次打印类型信息
                    print(f"Debug - next_batch type: {type(checkpoint_data['next_batch'])}")
                    print(f"Debug - total_correct type: {type(checkpoint_data['total_correct'])}")
                
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f, cls=NumpyEncoder)
                
                print(f"Checkpoint saved after batch {batch_idx + 1}")
            except Exception as e:
                print(f"Error saving checkpoint: {e}")
                # 尝试更原始的方式保存数据
                try:
                    with open(checkpoint_file, 'w') as f:
                        f.write(json.dumps({
                            'next_batch': int(batch_idx + 1),
                            'total_correct': int(total_rule_correct),
                            'partial_results': partial_file,
                            'timestamp': float(time.time())
                        }))
                    print("Checkpoint saved with manual type conversion")
                except Exception as e2:
                    print(f"Failed to save checkpoint even with manual conversion: {e2}")
    
    # 合并所有结果
    results_df = pd.concat(all_results, ignore_index=True)
    rule_correct_rate = total_rule_correct / len(results_df)
    
    # 保存结果
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    data_id = data_path.split("/")[-1].split(".")[0]
    output_file = os.path.join(output_path, f'{model_id}_{data_id}_2.csv')

    print(f'{model_id} {data_id} accuracy: {rule_correct_rate:.4f}')
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    # Clean up checkpoint file after successful completion
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    
    return rule_correct_rate

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='deepseek-chat')
    parser.add_argument('--data_path', type=str, default='./data/cleaned_tom/raw/Hi_ToM_cleaned.csv')
    parser.add_argument('--output_dir', type=str, default='./eval_tom/baseline_results')
    parser.add_argument('--batch_size', type=int, default=10, help='Number of examples to process in parallel')
    parser.add_argument('--checkpoint_interval', type=int, default=5, help='Save checkpoint every N batches')
    args = parser.parse_args()

    assert load_dotenv(find_dotenv(), override=True)

    model_id = args.model_path
    if model_id in ['gpt-4o-mini-2024-07-18', 'gpt-4o-2024-08-06', 'o3-mini-2025-01-31']:
        region = 'eastus'
    else:
        region = 'westus'

    if model_id == 'deepseek-chat':
        deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
        deepseek_base_url = os.getenv('DEEPSEEK_BASE_URL')
        client = AsyncOpenAI(
            api_key=deepseek_api_key,
            base_url=deepseek_base_url
        )
    else:
        openai_api_key = os.getenv('TONGGPT_API_KEY')
        openai_api_version = os.getenv('API_VERSION')
        openai_api_base = os.getenv('API_BASE')
        client = AsyncAzureOpenAI(
            api_key=openai_api_key,
            api_version=openai_api_version,
            azure_endpoint=f'{openai_api_base}/{region}'
        )

    asyncio.run(eval_model_async(client, model_id, args.data_path, args.output_dir, args.batch_size, args.checkpoint_interval))   