import argparse
import os
import re
import pandas as pd
from vllm import LLM, SamplingParams
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

model_path = 'Qwen/Qwen2.5-7B-Instruct-1M'

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = LLM(model=model_path, tokenizer=model_path, gpu_memory_utilization=0.9, tensor_parallel_size=2, max_model_len=4096)

sampling_params = SamplingParams(
    max_tokens=4096,
    temperature=0.6,
    top_k=-1,
    top_p=0.95,
)

def make_prompt(story, question):
    quiz = f"Read the following story and answer the question. \nStory: {story}\n Question: {question}"
    prefix = f"""<|im_start|>system\nYou are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. Now the user asks you to solve a theory of mind reasoning problem. After thinking, when you finally reach a conclusion, clearly state your answer within <answer> </answer> tags.\nNote: You should assume the following.\n(1) An agent witnesses everything and every movement before exiting a room.\n(2) An agent A can infer another agent B's mental state only if A and B have been in the same room, or have private or public interactions.\n<|im_end|>\n<|im_start|>user\n{quiz}\n<|im_end|>\n<|im_start|>assistant\n<think>"""
    # prompt = tokenizer.apply_chat_template(prefix, tokenize=False)
    return prefix

def make_inference(story, question):
    prompt = make_prompt(story, question)
    results = model.generate(prompt, sampling_params)
    return results[0]
