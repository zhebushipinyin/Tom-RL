import argparse
import os
import re
import pandas as pd
import asyncio
from tqdm.asyncio import tqdm_asyncio
from datasets import load_dataset
from openai import AzureOpenAI, AsyncAzureOpenAI
from dotenv import load_dotenv

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

async def call_openai_async(client, model_id, prompt):
    response = await client.chat.completions.create(
        model=model_id,
        messages=prompt,
        temperature=0.0,
        max_tokens=2048
    )
    return response.choices[0].message.content

def call_openai(client, model_id, prompt):
    response = client.chat.completions.create(
        model=model_id,
        messages=prompt,
        temperature=0.0,
        max_tokens=2048
    )
    return response.choices[0].message.content

async def eval_model_async(client, model_id, data_path, output_path, batch_size=50):
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
    
    # Debug模式，可以取少量数据测试
    # df = df.head(100)
    
    print(f"Processing {len(df)} examples in batches of {batch_size}")
    
    all_results = []
    total_rule_correct = 0
    total_batches = (len(df) + batch_size - 1) // batch_size
    
    # 批量处理
    for batch_idx in range(total_batches):
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
            
            cot_prompt = XML_COT_FORMAT.format(story, question)
            prompt = [{'role': 'system', 'content': SYSTEM_PROMPT}, 
                      {'role': 'user', 'content': cot_prompt}]
            prompts.append(prompt)
        
        # 并行处理所有样本
        tasks = [call_openai_async(client, model_id, prompt) for prompt in prompts]
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
        
        print(f"Batch accuracy: {batch_correct}/{len(batch_df)} = {batch_correct/len(batch_df):.4f}")
    
    # 合并所有结果
    results_df = pd.concat(all_results, ignore_index=True)
    rule_correct_rate = total_rule_correct / len(df)
    
    # 保存结果
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    data_id = data_path.split("/")[-1].split(".")[0]
    output_file = os.path.join(output_path, f'{model_id}_{data_id}.csv')

    print(f'{model_id} {data_id} accuracy: {rule_correct_rate:.4f}')
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    return rule_correct_rate

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='gpt-4o-mini-2024-07-18')
    parser.add_argument('--data_path', type=str, default='./data/cleaned_tom/raw/Hi_ToM_cleaned.csv')
    parser.add_argument('--output_dir', type=str, default='./eval_tom/baseline_results')
    parser.add_argument('--batch_size', type=int, default=50, help='Number of examples to process in parallel')
    args = parser.parse_args()

    load_dotenv()
    openai_api_key = os.getenv('TONGGPT_API_KEY')
    openai_api_version = os.getenv('API_VERSION')
    openai_api_base = os.getenv('API_BASE')

    model_id = args.model_path
    if model_id in ['gpt-4o-mini-2024-07-18', 'gpt-4o-2024-08-06', 'o3-mini-2025-01-31']:
        region = 'eastus'
    else:
        region = 'westus'

    client = AsyncAzureOpenAI(
        api_key=openai_api_key,
        api_version=openai_api_version,
        azure_endpoint=f'{openai_api_base}/{region}'
    )

    asyncio.run(eval_model_async(client, model_id, args.data_path, args.output_dir, args.batch_size))   