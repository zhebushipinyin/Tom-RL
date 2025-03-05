# train_grpo.py
import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
# from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
from typing import Dict, Tuple, Optional
# Load and prep dataset

SYSTEM_PROMPT = """You are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."""


# uncomment middle messages for 1-shot prompting
def get_questions() -> Dataset:
    data = load_dataset('parquet', data_files='data/ToM_KK_train.parquet')['train']
    #data = load_dataset('parquet', data_files='data/cleaned_train_ft_with_structure_384.parquet', split='train[:50%]')
    return data


dataset = get_questions()


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

def parse_solution_text_format(solution_text):
    """Parses ground truth solution text into status dictionary.
    
    Args:
        solution_text: Formatted solution text from dataset
        
    Returns:
        Dictionary mapping character names to their roles (knight/knave)
    """
    status_dict = {}
    
    for line in solution_text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        match = re.search(r'\b([A-Za-z]+)\b.*?\b(knight|knave)\b', line, re.IGNORECASE)
        if match:
            name, role = match.groups()
            status_dict[name] = role.lower()
    return status_dict


def parse_model_answer(answer_text, expected_names):
    status_dict = {}
    knight_count = answer_text.lower().count('knight')
    knave_count = answer_text.lower().count('knave')

    if knight_count + knave_count != len(expected_names):
        return None

    for name in expected_names:
        pattern = re.compile(
            rf'\b{re.escape(name)}\b\s+is\s+a\s+\b(knight|knave)\b', 
            re.IGNORECASE
        )
        match = pattern.search(answer_text) 
        if match:
            role = match.group(1).lower()
            status_dict[name] = role
        else:
            return None
    
    return status_dict


def reward_func_(response, answer, ability):
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    match = re.match(pattern, response, re.DOTALL | re.MULTILINE)

    if match:
        response_ = extract_xml_answer(response)
        norm_response = normalize_answer(response_)
        norm_answer = normalize_answer(answer)
        if ability!='logic':
            if norm_response == norm_answer:
                return 2
            else:
                return 0.5
        else:
            gt_status = parse_solution_text_format(norm_answer)
            expected_names = list(gt_status.keys())
            pred_status = parse_model_answer(norm_response, expected_names)
            if pred_status:
                if pred_status == gt_status:
                    return 2
                else:
                    # Paritial correct
                    return 0.8
            else:
                return 0.5
    else:
        return 0


def reward_function(completions, answer, ability, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    #print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extract_xml_answer(responses[0])}")
    print('-'*20, f"\nAbility:\n{ability[0]}",f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extract_xml_answer(responses[0])}")
    return [reward_func_(r, a, b) for r, a, b in zip(responses, answer, ability)]



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
    output_dir="outputs/Qwen-0.5B-GRPO-Tom-KK"
    run_name="Qwen-0.5B-GRPO-Tom-KK"

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
    gradient_accumulation_steps=8,
    num_generations=8,
    max_prompt_length=384,
    max_completion_length=512,
    num_train_epochs=1,
    save_steps=100,
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