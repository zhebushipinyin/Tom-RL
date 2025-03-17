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
    data = load_dataset('parquet', data_files='data/train/ToM_train_nt.parquet')['train']
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


def extract_xml_think(text: str) -> str:
    answer_pattern = r'<think>(.*?)</think>'
    matches = list(re.finditer(answer_pattern, text, re.DOTALL))
    if not matches:
        return ''
        
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




def reward_func_(response, answer):
    pattern = r".*?<answer>.*?</answer>$"
    
    tags = {
        'ans_start': ('<answer>', 1),
        'ans_end': ('</answer>', 1),
    }
    counts = 0
    for tag_name, (tag_str, expected_count) in tags.items():
        count = response.count(tag_str)
        if count == expected_count:
            counts +=1
    if counts == 2:
        match = re.match(pattern, response, re.DOTALL | re.MULTILINE)
        if match:
            response_ = extract_xml_answer(response)
            think = response.split('<answer>')[0]
            if len(think.split()) <= 10:
                return 0
            #len_reward = 0
            # if len(think.split())>20:
            #     len_reward = 0.2
            norm_response = normalize_answer(response_)
            norm_answer = normalize_answer(answer)
            #ans_pattern = r"\b(?:in|at|on|inside)?\s*(?:the\s*)?" + re.escape(norm_answer) + r"\b$"
            ans_pattern = r".*?" + re.escape(norm_answer) + r"\s*$"
            match = re.match(ans_pattern, norm_response, re.DOTALL | re.MULTILINE)
            if match:
                return 2
            else:
                norm_answer_ = norm_answer.replace('_', ' ')
                ans_pattern = r".*?" + re.escape(norm_answer_) + r"\s*$"
                match = re.match(ans_pattern, norm_response, re.DOTALL | re.MULTILINE)
                if match:
                    return 2
                return 0.5
    return 0


def reward_function(prompts, completions, answer, **kwargs) -> list[float]:
    q = prompts[0][-1]['content']
    responses = [completion[0]['content'] for completion in completions]
    
    #print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extract_xml_answer(responses[0])}")
    norm_answer = normalize_answer(answer[0])
    response_ = extract_xml_answer(responses[0])
    if response_ is None:
        response_ = 'None'
    print('-'*20,  
          f"\nAnswer:\n{norm_answer}", 
          f"\nResponse:\n{responses[0]}", 
          f"\nExtracted:\n{response_}")
    return [reward_func_(r, a) for r, a in zip(responses, answer)]


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
    output_dir="outputs/Qwen-0.5B-GRPO-HI-NT"
    run_name="Qwen-0.5B-GRPO-HI-NT"

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
    max_prompt_length=512,
    max_completion_length=512,
    num_train_epochs=2,
    save_steps=100,
    max_grad_norm=0.1,
    report_to="wandb",
    log_on_each_node=False,
    gradient_checkpointing=True,
    beta=0.001,
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