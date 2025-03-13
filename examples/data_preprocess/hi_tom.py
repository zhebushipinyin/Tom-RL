"""
Preprocess the Facebook/ExploreToM dataset to parquet format
"""

import os
import datasets
import argparse

import pandas as pd
from verl.utils.hdfs_io import copy, makedirs


def make_prefix(story, question, template_type='base', add_hint=False, wo_think=False):
    """
    Format the prompt with appropriate instructions based on template type.
    
    Args:
        story: The infilled story text
        question: The question about the story
        template_type: The template format to use ('base' or 'qwen-instruct')
        
    Returns:
        Formatted prompt with instructions
    """
    quiz = f"Read the following story and answer the question. \nStory: {story}\n Question: {question}"
    # quiz = story + "\n\n" + question
    
    if template_type == 'base':
        prefix = f"""The user asks a question about a story, and the Assistant answers it. The assistant first thinks about the reasoning process in the mind and then provides the user with the final answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. Now the user asks you to solve a theory of mind reasoning problem. After thinking, when you finally reach a conclusion, clearly state your answer within <answer> </answer> tags.\n\nUser:{quiz}\nAssistant: <think>"""
    # TODO add hints
    elif template_type == 'qwen-instruct':
        if not add_hint:
            if not wo_think:
                prefix = f"""<|im_start|>system\nYou are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. Now the user asks you to solve a theory of mind reasoning problem. After thinking, when you finally reach a conclusion, clearly state your answer within <answer> </answer> tags.\n<|im_end|>\n<|im_start|>user\n{quiz}\n<|im_end|>\n<|im_start|>assistant\n<think>"""
            else:
                prefix = f"""<|im_start|>system\nYou are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. Now the user asks you to solve a theory of mind reasoning problem. Please reason step by step, and put your final answer within <answer> </answer> tags.\n<|im_end|>\n<|im_start|>user\n{quiz}\n<|im_end|>\n<|im_start|>assistant\n"""
        else:
            if not wo_think:
                prefix = f"""<|im_start|>system\nYou are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. Now the user asks you to solve a theory of mind reasoning problem. After thinking, when you finally reach a conclusion, clearly state your answer within <answer> </answer> tags.\nNote: You should assume the following.\n(1) An agent witnesses everything and every movement before exiting a room.\n(2) An agent A can infer another agent B's mental state only if A and B have been in the same room, or have private or public interactions.\n<|im_end|>\n<|im_start|>user\n{quiz}\n<|im_end|>\n<|im_start|>assistant\n<think>"""
            else:
                prefix = f"""<|im_start|>system\nYou are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. Now the user asks you to solve a theory of mind reasoning problem. Please reason step by step, and put your final answer within <answer> </answer> tags.\nNote: You should assume the following.\n(1) An agent witnesses everything and every movement before exiting a room.\n(2) An agent A can infer another agent B's mental state only if A and B have been in the same room, or have private or public interactions.\n<|im_end|>\n<|im_start|>user\n{quiz}\n<|im_end|>\n<|im_start|>assistant\n"""
    elif template_type == 'dpsk-reasoning':
        prefix = "<｜begin▁of▁sentence｜><｜User｜>You are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. Now the user asks you to solve a theory of mind reasoning problem. After thinking, when you finally reach a conclusion, clearly state your answer within <answer> </answer> tags.<｜Assistant｜><think>"
    return prefix


def explore_tom_make_map_fn(data_source, story_type='story_structure'):
    def process_fn(example, idx):
        # Choose which story to use based on story_type parameter
        story = example['story_structure'] if story_type == 'story_structure' else example['infilled_story']
        
        # Create data point using the selected story
        prompt = make_prefix(
            story=story, 
            question=example['question'],
            template_type=args.template_type,
            add_hint=args.add_hint,
            wo_think=args.wo_think
        )
        
        # Get the expected answer as ground truth
        solution = example['answer']
        # Create extra_info dictionary with all the other fields
        extra_info = {
            'story_structure': example['story_structure'],
            'infilled_story': example['infilled_story'],
            'story_type': example['story_type'],
            'question_type': example['question_type'],
            'params': example['params'],
            'nth_order': example['nth_order'],
            'is_interesting': example['is_interesting'],
        }
        
        return {
            "data_source": data_source,
            "prompt": [{
                "role": "user",
                "content": prompt,
            }],
            "ability": "theory_of_mind",
            "reward_model": {
                "style": "rule",
                "ground_truth": solution
            },
            "extra_info": extra_info
        }
    return process_fn

def hi_tom_make_map_fn(data_source):
    def process_fn(example, idx):
        question = make_prefix(
            story=example['story'], 
            question=example['question_old'],
            template_type=args.template_type,
            add_hint=args.add_hint,
            wo_think=args.wo_think
        )
        solution = example['answer']
        return {
            "data_source": data_source,
            "prompt": [{
                "role": "user",
                "content": question,
            }],
            "ability": "theory_of_mind",
            "reward_model": {
                "style": "rule",
                "ground_truth": solution
            },
            "extra_info": {'deception': example['deception'], 
                            'story_length': example['story_length'],
                            'question_order': example['question_order'],
                            }
        }
    return process_fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/cleaned_tom')
    parser.add_argument('--hdfs_dir', default=None)
    # random seed
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--template_type', type=str, default='qwen-instruct', 
                        choices=['base', 'qwen-instruct', 'dpsk-reasoning'])
    parser.add_argument('--add_hint', action='store_true', 
                        help='Add hint to the prompt')
    parser.add_argument('--wo_think', action='store_true', 
                        help='Whether to remove the thinking tags')
    args = parser.parse_args()

    # deception,story_length,question_order,sample_id,story,question,choices,answer,question_old,answer_old
    data_source = './data/cleaned_tom/raw/hi_tom_3000.csv'
    explore_tom_source = './data/cleaned_tom/raw/ToM_train_600.parquet'

    explore_tom_dataset = datasets.load_dataset('parquet', data_files=explore_tom_source)['train']

    # 按照 question_order 拆分数据，question_order 为 0-1-2-3-4
    # order 为 4 的为test set（600条），其余的2400条，拆出400条，也作为测试集，这样训练集 2000 条，测试集 1000 条
    dataset = pd.read_csv(data_source)
    train_dataset = dataset[dataset['question_order'] != 4]
    test_dataset = dataset[dataset['question_order'] == 4].reset_index(drop=True)
    train_temp = train_dataset.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    additional_test = train_temp.iloc[-400:].reset_index(drop=True)
    train_temp = train_temp.iloc[:2000].reset_index(drop=True)
    test_dataset_combined = pd.concat([test_dataset, additional_test], ignore_index=True)

    train_dataset = datasets.Dataset.from_pandas(train_temp)
    test_dataset = datasets.Dataset.from_pandas(test_dataset_combined)

    print('len of train_dataset:', len(train_dataset))
    print('len of test_dataset (hi_tom):', len(test_dataset))

    local_dir = os.path.expanduser(args.local_dir)
    hdfs_dir = args.hdfs_dir
    
    train_dataset = train_dataset.map(function=hi_tom_make_map_fn('hi_tom'), with_indices=True)   

    test_dataset = test_dataset.map(function=hi_tom_make_map_fn('hi_tom'), with_indices=True)
    explore_tom_dataset = explore_tom_dataset.map(function=explore_tom_make_map_fn('explore_tom'), with_indices=True)
    # 将 explore_tom_dataset 和 test_dataset 合并
    test_dataset = datasets.concatenate_datasets([test_dataset, explore_tom_dataset])
    print('len of test_dataset (hi_tom + explore_tom):', len(test_dataset))

    if args.add_hint:
        if args.wo_think:
            train_dataset.to_parquet(os.path.join(local_dir, 'hi_tom_train_2000_hint_wo_think.parquet'))
            test_dataset.to_parquet(os.path.join(local_dir, 'hi_tom_explore_tom_test_hint_wo_think.parquet'))
        else:
            train_dataset.to_parquet(os.path.join(local_dir, 'hi_tom_train_2000_hint.parquet'))
            test_dataset.to_parquet(os.path.join(local_dir, 'hi_tom_explore_tom_test_hint.parquet'))
    else:
        if args.wo_think:
            train_dataset.to_parquet(os.path.join(local_dir, 'hi_tom_train_2000_wo_think.parquet'))
            test_dataset.to_parquet(os.path.join(local_dir, 'hi_tom_explore_tom_test_wo_think.parquet'))
        else:
            train_dataset.to_parquet(os.path.join(local_dir, 'hi_tom_train_2000.parquet'))
            test_dataset.to_parquet(os.path.join(local_dir, 'hi_tom_explore_tom_test.parquet'))

    # Create local directory if not exists
    os.makedirs(local_dir, exist_ok=True)

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
