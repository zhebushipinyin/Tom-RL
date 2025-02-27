"""
Preprocess the Facebook/ExploreToM dataset to parquet format
"""

import os
import datasets
import argparse
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
    elif template_type == 'qwen-instruct':
        prefix = f"""<|im_start|>system\nYou are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. Now the user asks you to solve a theory of mind reasoning problem. After thinking, when you finally reach a conclusion, clearly state your answer within <answer> </answer> tags.\n<|im_end|>\n<|im_start|>user\n{quiz}\n<|im_end|>\n<|im_start|>assistant\n<think>"""
    
    return prefix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/explore_tom')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, default='qwen-instruct', choices=['base', 'qwen-instruct'])

    args = parser.parse_args()

    data_source = 'facebook/ExploreToM'

    # Load the dataset
    dataset = datasets.load_dataset(data_source)
    
    # Since ExploreToM only has a train split, we'll split it 80/20
    train_test_split = dataset['train'].train_test_split(test_size=0.2, seed=42)
    train_dataset = train_test_split['train']
    test_dataset = train_test_split['test']

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            # Format the prompt using make_prefix
            question = make_prefix(
                story=example['infilled_story'], 
                question=example['question'],
                template_type=args.template_type
            )
            
            # Get the expected answer as ground truth
            solution = example['expected_answer']
            
            # Create extra_info dictionary with all the other fields
            extra_info = {
                'split': split,
                'index': idx,
                'story_structure': example['story_structure'],
                'qprop=params': example['qprop=params'],
                'qprop=nth_order': example['qprop=nth_order'],
                'qprop=non_unique_mental_state': example['qprop=non_unique_mental_state'],
                'sprop=is_false_belief_story_1st': example['sprop=is_false_belief_story_1st'],
                'sprop=is_false_belief_story_1st_and_2nd': example['sprop=is_false_belief_story_1st_and_2nd'],
                'sprop=story_accuracy_1st_raw': example['sprop=story_accuracy_1st_raw'],
                'sprop=story_accuracy_1st_infilled': example['sprop=story_accuracy_1st_infilled'],
                'sprop=global_idx': example['sprop=global_idx'],
                'param=story_type': example['param=story_type'],
                'param=num_stories_total': example['param=num_stories_total'],
                'param=max_sentences': example['param=max_sentences'],
                'param=num_people': example['param=num_people'],
                'param=num_moves': example['param=num_moves'],
                'param=num_rooms': example['param=num_rooms']
            }
            
            data = {
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
                "extra_info": extra_info
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = os.path.expanduser(args.local_dir)
    hdfs_dir = args.hdfs_dir

    # Create local directory if not exists
    os.makedirs(local_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
