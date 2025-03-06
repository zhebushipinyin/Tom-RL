# train_grpo.py
import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
# from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

# Load and prep dataset


# uncomment middle messages for 1-shot prompting
def get_questions() -> Dataset:
    data = load_dataset('parquet', data_files='data/train/ToM_train_multiquestion.parquet')['train']
    return data


dataset = get_questions()
#testset = get_gsm8k_questions('test')


def extract_xml_answer(text: str) -> str:
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, text, re.DOTALL))
    if not matches:
        return None
        
    final_answer = matches[-1].group(1).strip()
    return final_answer



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


def extract_ordered_answers(text, correct_answers):
    """
    按照指定顺序从文本中提取正确答案
    
    Args:
        text (str): 输入文本
        correct_answers (list): 预期正确答案的列表（按顺序）
    
    Returns:
        list: 解析出的答案列表，按正确答案的顺序排列
    """
    # 按“。；\n”分割文本为句子列表
    sentences = re.split(r"[。；;.\n]", text)
    sentences = [normalize_answer(s) for s in sentences]
    extracted_answers = []
    last_matched_index = -1  # 记录上一次匹配成功的索引，确保匹配顺序
    
    for correct_ai in correct_answers:
        pattern = r"\b(?:in|at|on|inside)?\s*(?:the\s*)?" + re.escape(correct_ai) + r"\b"

        # 只从上一次匹配成功的位置往后搜索
        for i in range(last_matched_index + 1, len(sentences)):
            if re.search(pattern, sentences[i], re.IGNORECASE):
                extracted_answers.append(correct_ai)
                last_matched_index = i  # 记录匹配成功的索引
                break  # 只允许匹配一次，防止重复使用同一部分文本
    return extracted_answers


def reward_func_(response, answer):
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    match = re.match(pattern, response, re.DOTALL | re.MULTILINE)

    if match:
        response_ = extract_xml_answer(response)
        if response_ is None:
            return 0.5
        norm_response = response_
        norm_answer = [normalize_answer(each) for each in answer]
        pred_status = extract_ordered_answers(norm_response, norm_answer)
        if pred_status == norm_answer:
            return 2
        else:
            return 0.5 + 0.05*len(pred_status)
    else:
        return 0


def reward_function(prompts, completions, answer, **kwargs) -> list[float]:
    q = prompts[0][-1]['content']
    responses = [completion[0]['content'] for completion in completions]
    
    #print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extract_xml_answer(responses[0])}")
    norm_answer = [normalize_answer(each) for each in answer[0]]
    response_ = extract_xml_answer(responses[0])
    if response_ is None:
        response_ = 'None'
    print('-'*20, 
          f"\Question:\n{q[:500]}", 
          f"\nAnswer:\n{norm_answer}", 
          f"\nResponse:\n{responses[0]}", 
          f"\nExtracted:\n{response_}",
          f"\nExtracted Answer:\n{extract_ordered_answers(response_, norm_answer)}")
    return [reward_func_(r, a) for r, a in zip(responses, answer)]


def format_reward(completions, **kwargs):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1 if match else 0.0 for match in matches]



#model_name = "meta-llama/Llama-3.2-1B-Instruct"
#model_name = "Qwen/Qwen2.5-1.5B-Instruct"
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

if "Llama" in model_name:
    output_dir = "outputs/Llama-1B-GRPO"
    run_name = "Llama-1B-GRPO-gsm8k"
elif model_name =='Qwen2.5-1.5B-Instruct':
    output_dir="outputs/Qwen-1.5B-GRPO"
    run_name="Qwen-1.5B-GRPO-gsm8k"
else:
    output_dir="outputs/Qwen-0.5B-GRPO-ToM-MQ"
    run_name="Qwen-0.5B-GRPO-ToM-MQ"

training_args = GRPOConfig(
    output_dir=output_dir,
    run_name=run_name,
    learning_rate=5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type='cosine',
    logging_steps=1,
    bf16=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_generations=8,
    max_prompt_length=512,
    max_completion_length=512,
    num_train_epochs=3,
    save_steps=50,
    max_grad_norm=0.1,
    report_to="wandb",
    log_on_each_node=False,
    gradient_checkpointing=True,
)

"""
peft_config = LoraConfig(
    r=16,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    task_type="CAUSAL_LM",
    lora_dropout=0.05,
)"""

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map=None
).to("cuda")
        
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# use peft at your own risk; not working for me with multi-GPU training
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[reward_function],
    args=training_args,
    train_dataset=dataset,
    #peft_config=peft_config
)
trainer.train()