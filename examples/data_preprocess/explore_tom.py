"""
Preprocess the Facebook/ExploreToM dataset to parquet format
"""

import os
import datasets
import argparse

import pandas as pd
from verl.utils.hdfs_io import copy, makedirs


def make_prefix(story, question, template_type='base'):
    """
    Format the prompt with appropriate instructions based on template type.
    
    Args:
        story: The infilled story text
        question: The question about the story
        template_type: The template format to use ('base' or 'qwen-instruct')
        
    Returns:
        Formatted prompt with instructions
    """
    quiz = story + "\n\n" + question
    
    if template_type == 'base':
        prefix = f"""The user asks a question about a story, and the Assistant answers it. The assistant first thinks about the reasoning process in the mind and then provides the user with the final answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. Now the user asks you to solve a theory of mind reasoning problem. After thinking, when you finally reach a conclusion, clearly state your answer within <answer> </answer> tags.\n\nUser:{quiz}\nAssistant: <think>"""
    # TODO add hints
    elif template_type == 'qwen-instruct':
        prefix = f"""<|im_start|>system\nYou are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. Now the user asks you to solve a theory of mind reasoning problem. After thinking, when you finally reach a conclusion, clearly state your answer within <answer> </answer> tags.\n<|im_end|>\n<|im_start|>user\n{quiz}\n<|im_end|>\n<|im_start|>assistant\n<think>"""
    elif template_type == 'dpsk-reasoning':
        prefix = "<｜begin▁of▁sentence｜><｜User｜>You are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. Now the user asks you to solve a theory of mind reasoning problem. After thinking, when you finally reach a conclusion, clearly state your answer within <answer> </answer> tags.<｜Assistant｜><think>"
    return prefix


def train_make_map_fn(data_source, story_type='story_structure'):
    def process_fn(example, idx):
        # Choose which story to use based on story_type parameter
        story = example['story_structure'] if story_type == 'story_structure' else example['infilled_story']
        
        # Create data point using the selected story
        prompt = make_prefix(
            story=story, 
            question=example['question'],
            template_type=args.template_type
        )
        
        # Get the expected answer as ground truth
        solution = example['answer']
        # Create extra_info dictionary with all the other fields
        extra_info = {
            'index': idx,
            'story_structure': example['story_structure'],
            'infilled_story': example['infilled_story'],
            'story_type': example['story_type'],
            'story_structure_wn': example['story_structure_wn'],
            'question_type': example['question_type'],
            'answer': example['answer'],
            'params': example['params'],
            'nth_order': example['nth_order'],
            'is_interesting': example['is_interesting'],
            'story_used': story_type
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

def test_make_map_fn(data_source):
    def process_fn(example, idx):
        question = make_prefix(
            story=example['story'], 
            question=example['question_old'],
            template_type=args.template_type
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
    parser.add_argument('--use_both_stories', action='store_true', 
                        help='Generate two data points for each example, one using story_structure and one using infilled_story')

    args = parser.parse_args()

    data_source = './data/cleaned_tom/raw/ToM_train_600.parquet'
    train_dataset = pd.read_parquet(data_source)
    train_dataset = datasets.Dataset.from_pandas(train_dataset)
    test_source = './data/cleaned_tom/raw/Hi_ToM_cleaned.csv'
    test_dataset = datasets.load_dataset('csv', data_files=test_source)['train']

    
    local_dir = os.path.expanduser(args.local_dir)
    hdfs_dir = args.hdfs_dir
    
    # Process the datasets
    if args.use_both_stories:
        # Create two separate datasets - one for each story type
        train_dataset_struct = train_dataset.map(
            function=train_make_map_fn('explore_tom', 'story_structure'), 
            with_indices=True
        )
        train_dataset_infill = train_dataset.map(
            function=train_make_map_fn('explore_tom', 'infilled_story'), 
            with_indices=True
        )
        
        # Concatenate the two datasets
        train_dataset = datasets.concatenate_datasets([train_dataset_struct, train_dataset_infill])
        
        # Shuffle the combined dataset
        train_dataset = train_dataset.shuffle(seed=args.seed)
        print('len of train_dataset (both stories):', len(train_dataset))
        train_dataset.to_parquet(os.path.join(local_dir, 'explore_tom_train_600_both.parquet'))
    else:
        train_dataset = train_dataset.map(
            function=train_make_map_fn('explore_tom', 'story_structure'), 
            with_indices=True
        )
        print('len of train_dataset:', len(train_dataset))
        train_dataset.to_parquet(os.path.join(local_dir, 'explore_tom_train_600.parquet'))
    
    test_dataset = test_dataset.map(function=test_make_map_fn('hi_tom'), with_indices=True)
    test_dataset.to_parquet(os.path.join(local_dir, 'hi_tom_test.parquet'))


    # Create local directory if not exists
    os.makedirs(local_dir, exist_ok=True)

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
