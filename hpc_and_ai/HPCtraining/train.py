# pip install -U torch transformers accelerate peft trl datasets bitsandbytes

# python -m pip install -U pip
# python -m pip install -U bitsandbytes

import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer


model_id = "google/gemma-3-1b-it"

# huggingface token
hf_token = "hf_"

"""
hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
if not hf_token:
    raise RuntimeError(
        "Windows (PowerShell): $env:HF_TOKEN='hf_xxx'"
    )
"""

"""
# QLoRA (4-bit) to save VRAM, quantinize model first and then train quantinized model (memory efficency turning training)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # quantinizing format
    bnb_4bit_compute_dtype=torch.bfloat16,  # original format
)
"""

tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=hf_token, trust_remote_code=True)

# ensure pad token exists
if tokenizer.pad_token_id is None:
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    added_tokens = True
else:
    added_tokens = False

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    use_auth_token=hf_token,
    trust_remote_code=True,
)

# Important for training with gradient checkpointing / stability
model.config.use_cache = False

# required for k-bit training (4-bit / 8-bit)
#model = prepare_model_for_kbit_training(model)

# PEFT LoRA config
lora_config = LoraConfig(
    r=8,  # rank, how many directions the model is allowed to adjust in each weight matrix
    lora_alpha=16,  # 16/8 = alpha/rank, lora influence
    lora_dropout=0.05,  # The dropout probability for Lora layers, avoid overfit. (1 - treeningandmetele kohandumise %)
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # change focus {choose word} (word weights, word relations to other questions)
        "gate_proj", "up_proj", "down_proj",  # create word patterns {use word}
    ],
)

# wrap model with LoRA
model = get_peft_model(model, lora_config)

# training dataset (json)
# CHANGE FILENAME!!
with open("facts.json", "r", encoding="utf-8") as f:
    examples = json.load(f)

train_ds = Dataset.from_list(examples)

# model's chat template
def format_example(ex):
    messages = [
        {
            "role": "system",
            "content": "Sa oled abivalmis assistent. Vasta lühidalt ja eesti keeles."
        },
        {"role": "user", "content": ex["question"]},
        {"role": "assistant", "content": ex["answer"]},
    ]
    # This returns a single string containing the properly templated chat transcript.
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}

train_ds = train_ds.map(format_example)

# training arguments
args = TrainingArguments(
    output_dir="checkpoint_lora",  # saves training info, checkpoints
    per_device_train_batch_size=1,  # 1 example on 1 gpu at a time (8gb gpu = 1, 12-16gb = 1-2, 16gb+ = 2-4)
    gradient_accumulation_steps=4,  # how many batches are collected, before gradient update, noise
    learning_rate=1e-5,  # training speed
    num_train_epochs=1,  # how many times dataset is looped over, SLIGHT CHANCE OF OVERFIT WITH OUR DATASETS
    logging_steps=1,  # log loss after each step
    save_steps=50,  # save checkpoint
    #fp16=False,  # for float16 GPUs
    bf16=torch.cuda.is_available(),  # bfloat16 (better if supported)
    report_to="none",  # visualize training curve ("tensorboard", "wandb")
)

# training configs
trainer = SFTTrainer(
    model=model,
    train_dataset=train_ds,
    peft_config=lora_config,
    args=args,
    dataset_text_field="text",
    max_seq_length=512,  # max token len
)

# start the training
trainer.train()

# Save LoRA adapter configs
model.save_pretrained("gemma_1.0_lora")
tokenizer.save_pretrained("gemma_1.0_lora")

print("Done. Saved LoRA adapter to: gemma_1.0_lora")
