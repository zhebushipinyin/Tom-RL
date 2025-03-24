import argparse
import os
import re
import pandas as pd
from vllm import LLM, SamplingParams
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

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


def eval_model(model_path, data_path, output_path, tp):
    
    model = LLM(model=model_path, tokenizer=model_path, 
                gpu_memory_utilization=0.9, tensor_parallel_size=tp, 
                max_model_len=4096)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    sampling_params = SamplingParams(
        max_tokens=2048,
        temperature=0.6,
        top_k=-1,
        top_p=0.95,
    )
    if data_path.endswith('.parquet'):
        # ds = pd.read_parquet(data_path)
        ds = load_dataset('parquet', data_files=data_path)['train']
    elif data_path.endswith('.csv'):
        # ds = pd.read_csv(data_path)
        # ds = Dataset.from_pandas(ds)
        ds = load_dataset('csv', data_files=data_path)['train']
    else:
        ds = load_dataset(data_path)
    
    eval_prompts = []

    for example in ds:

        if 'story' in example:
            # hi_tom
            story = example['story']
            if 'question_old' in example:
                question = example['question_old']
            else:
                question = example['question']
        else:
            # explore_tom
            story = example['story_structure']
            question = example['question']
        
        if 'data_source' in example and example['data_source'] == 'explore_tom':
            story = example['extra_info']['infilled_story']
            question = example['question']


        cot_prompt = XML_COT_FORMAT.format(story, question)
        prompt = [{'role': 'system', 'content': SYSTEM_PROMPT}, 
                  {'role': 'user', 'content': cot_prompt}]
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        eval_prompts.append(prompt)
    
    model_results = model.generate(eval_prompts, sampling_params, use_tqdm=True)
    results = []
    rule_correct = 0
    for example, result in zip(ds, model_results):
        model_answer = result.outputs[0].text
        example['model_answer'] = model_answer
        gt_answer = example['answer'] if 'answer' in example else example['expected_answer']
        example['rule_based_eval'] = reward_func(model_answer, gt_answer)
        results.append(example)
        rule_correct += example['rule_based_eval']
    rule_correct_rate = rule_correct / len(ds)
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    model_id = model_path.split('/')[-1]
    data_id = data_path.split("/")[-1].split(".")[0]

    print(f'{model_id} {data_id} rule_correct_rate: {rule_correct_rate}')

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path + '/' + f'{model_id}_{data_id}_2.csv', index=False, encoding='utf-8-sig')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='Qwen/Qwen2.5-1.5B-Instruct')
    parser.add_argument('--tp', type=int, default=2)
    # parser.add_argument('--data_path', type=str, default='./data/cleaned_tom/raw/hi_tom_3000.csv')
    parser.add_argument('--data_path', type=str, default='./data/cleaned_tom/raw/ToM_train_600.parquet')
    parser.add_argument('--output_dir', type=str, default='./eval_tom/basline')
    args = parser.parse_args()
    eval_model(args.model_path, args.data_path, args.output_dir, args.tp)   