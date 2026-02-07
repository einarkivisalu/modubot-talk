# pip install -U transformers accelerate peft trl datasets bitsandbytes

import os
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer

# -----------------------------
# 1) Choose a Gemma chat model
# -----------------------------
# Use a Gemma *instruction/chat* model for chat-style prompts.
# If you use gemma-3-4b-it, keep in mind training needs a GPU.
model_id = "google/gemma-3-4b-it"

# If the model is gated, you may need:
# export HF_TOKEN=...
hf_token = os.environ.get("HF_TOKEN", None)

# -----------------------------
# 2) Optional: QLoRA (4-bit) to save VRAM
# -----------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=hf_token,
    device_map="auto",
    quantization_config=bnb_config,  # remove this line if doing FP16/BF16 LoRA
)

# Important for training with gradient checkpointing / stability
model.config.use_cache = False

# -----------------------------
# 3) PEFT LoRA configuration
# -----------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

# -----------------------------
# 4) Tiny training dataset
#    (You should add MANY variants, not just one line.)
# -----------------------------
examples = [
    {
        "question": "Why is the sky blue?",
        "answer": "The sky is blue because of the atmosphere."
    },
    # Add paraphrases to make it robust:
    {
        "question": "Why does the sky look blue?",
        "answer": "The sky is blue because of the atmosphere."
    },
    {
        "question": "Explain why the sky is blue.",
        "answer": "The sky is blue because of the atmosphere."
    },
]

train_ds = Dataset.from_list(examples)

# -----------------------------
# 5) Format into the model’s chat template
# -----------------------------
def format_example(ex):
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer briefly."},
        {"role": "user", "content": ex["question"]},
        {"role": "assistant", "content": ex["answer"]},
    ]
    # This returns a single string containing the properly templated chat transcript.
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}

train_ds = train_ds.map(format_example)

# -----------------------------
# 6) Train with TRL SFTTrainer
# -----------------------------
args = TrainingArguments(
    output_dir="gemma_skyblue_lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=1,
    save_steps=50,
    fp16=False,                 # for bfloat16 GPUs set bf16=True instead
    bf16=torch.cuda.is_available(),  # good default on modern GPUs
    optim="paged_adamw_8bit",   # good with QLoRA
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_ds,
    peft_config=lora_config,
    args=args,
    dataset_text_field="text",
    max_seq_length=512,
)

trainer.train()

# Save LoRA adapter
trainer.model.save_pretrained("gemma_skyblue_lora_adapter")
tokenizer.save_pretrained("gemma_skyblue_lora_adapter")

print("Done. Saved LoRA adapter to: gemma_skyblue_lora_adapter")

# -----------------------------
# 7) Test (in the same script)
# -----------------------------
def chat(prompt: str, max_new_tokens: int = 50):
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer briefly."},
        {"role": "user", "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        out = trainer.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    # Decode only the new tokens
    new_tokens = out[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

print(chat("Why is the sky blue?"))
