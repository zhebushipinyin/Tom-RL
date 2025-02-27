# Analysis of Problem Datasets Across Difficulty Configurations

## Problem Counts

| Difficulty | Train Problems | Test Problems | Total Problems |
|------------|---------------|--------------|---------------|
| 3ppl | 900 | 100 | 1000 |
| 4ppl | 900 | 100 | 1000 |
| 5ppl | 900 | 100 | 1000 |
| 6ppl | 900 | 100 | 1000 |
| 7ppl | 900 | 100 | 1000 |

## Attribute Comparison

### Train Dataset Attributes

All train datasets across different difficulty configurations have the same attributes.

**Attributes:**

- quiz
- names
- knight_knave
- solution
- solution_text
- solution_text_format
- cot_head
- cot_repeat_steps
- cot_foot
- statements
- index
- data_source
- prompt
- ability
- reward_model
- extra_info

### Test Dataset Attributes

All test datasets across different difficulty configurations have the same attributes.

**Attributes:**

- quiz
- names
- knight_knave
- solution
- solution_text
- solution_text_format
- cot_head
- cot_repeat_steps
- cot_foot
- statements
- index
- data_source
- prompt
- ability
- reward_model
- extra_info

## Sample from 5ppl Test Dataset

Below is a sample problem from the 5ppl test dataset:

```json
{
  "quiz": "A very special island is inhabited only by knights and knaves. Knights always tell the truth, and knaves always lie. You meet 5 inhabitants: Ella, Zoey, Scarlett, Henry, and Amelia. Ella expressed that Zoey is a knight or Zoey is a knave. In a statement by Zoey: \"Scarlett is a knave\". Scarlett commented, \"Ella is a knave or Amelia is a knave\". \"Scarlett is a knight or Amelia is a knight,\" Henry declared. According to Amelia, \"Henry is a knave\". So who is a knight and who is a knave?",
  "names": [
    "Ella",
    "Zoey",
    "Scarlett",
    "Henry",
    "Amelia"
  ],
  "knight_knave": {
    "Knave": "Knave",
    "Knight": "Knight",
    "a_knave": "a knave",
    "a_knight": "a knight",
    "knave": "knave",
    "knight": "knight"
  },
  "statements": "(('or', ('telling-truth', 1), ('lying', 1)), ('lying', 2), ('or', ('lying', 0), ('lying', 4)), ('or', ('telling-truth', 2), ('telling-truth', 4)), ('lying', 3))",
  "solution": [
    true,
    false,
    true,
    true,
    false
  ],
  "solution_text": "Ella is a knight, Zoey is a knave, Scarlett is a knight, Henry is a knight, and Amelia is a knave.",
  "solution_text_format": "(1) Ella is a knight\n(2) Zoey is a knave\n(3) Scarlett is a knight\n(4) Henry is a knight\n(5) Amelia is a knave",
  "cot_head": "Let's think step by step, by considering whether each person is lying and if that leads to contradiction.",
  "cot_repeat_steps": [
    "Assume Ella is a knight. No contradiction is found in their claim that Zoey is a knight or Zoey is a knave.",
    "Assume Zoey is a knight. No contradiction is found in their claim that Scarlett is a knave.",
    "Scarlett cannot be a knight, because this would contradict the claim of Zoey that Scarlett is a knave.",
    "Assume Scarlett is a knave. No contradiction is found in their false claim that Ella is a knave or Amelia is a knave.",
    "Assume Amelia is a knight. No contradiction is found in their claim that Henry is a knave.",
    "Henry cannot be a knight, because this would contradict the claim of Amelia that Henry is a knave.",
    "Henry cannot be a knave, because this would contradict the false claim of their own that Scarlett is a knight or Amelia is a knight.",
    "We have exhausted all possibilities for Henry, so let us go back and reconsider Amelia.",
    "Amelia cannot be a knave, because this would contradict the false claim of Scarlett that Ella is a knave or Amelia is a knave.",
    "We have exhausted all possibilities for Amelia and Scarlett, so let us go back and reconsider Zoey.",
    "Assume Zoey is a knave. No contradiction is found in their false claim that Scarlett is a knave.",
    "Assume Scarlett is a knight. No contradiction is found in their claim that Ella is a knave or Amelia is a knave.",
    "Amelia cannot be a knight, because this would contradict the claim of Scarlett that Ella is a knave or Amelia is a knave.",
    "Assume Amelia is a knave. No contradiction is found in their false claim that Henry is a knave.",
    "Assume Henry is a knight. No contradiction is found in their claim that Scarlett is a knight or Amelia is a knight."
  ],
  "cot_foot": "This leads to a feasible solution.",
  "index": 1000,
  "data_source": "kk_logic",
  "prompt": [
    {
      "content": "<|im_start|>system\nYou are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.  Now the user asks you to solve a logical reasoning problem. After thinking, when you finally reach a conclusion, clearly state the identity of each character within <answer> </answer> tags. i.e., <answer> (1) Zoey is a knight\n(2) ... </answer>.\n<|im_end|>\n<|im_start|>user\nA very special island is inhabited only by knights and knaves. Knights always tell the truth, and knaves always lie. You meet 5 inhabitants: Ella, Zoey, Scarlett, Henry, and Amelia. Ella expressed that Zoey is a knight or Zoey is a knave. In a statement by Zoey: \"Scarlett is a knave\". Scarlett commented, \"Ella is a knave or Amelia is a knave\". \"Scarlett is a knight or Amelia is a knight,\" Henry declared. According to Amelia, \"Henry is a knave\". So who is a knight and who is a knave?\n<|im_end|>\n<|im_start|>assistant\n<think>",
      "role": "user"
    }
  ],
  "ability": "logic",
  "reward_model": {
    "ground_truth": {
      "solution_text_format": "(1) Ella is a knight\n(2) Zoey is a knave\n(3) Scarlett is a knight\n(4) Henry is a knight\n(5) Amelia is a knave",
      "statements": "(('or', ('telling-truth', 1), ('lying', 1)), ('lying', 2), ('or', ('lying', 0), ('lying', 4)), ('or', ('telling-truth', 2), ('telling-truth', 4)), ('lying', 3))"
    },
    "style": "rule"
  },
  "extra_info": {
    "index": 0,
    "split": "test"
  }
}
```
