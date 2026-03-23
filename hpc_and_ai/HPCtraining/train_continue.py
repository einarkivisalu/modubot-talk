# train_continue.py

import os
import json
import inspect
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from huggingface_hub import login

# ------- Config -------
model_id = "google/gemma-3-1b-it"
facts_path = "luuletused.json"
output_adapter_dir = "gemma_1.0_lora"
checkpoint_dir = "checkpoint_lora"

# Set this to True to keep training from an already-saved LoRA adapter.
# Set to False to start a brand-new adapter training run.
load_existing_adapter = True

# ------- HF login -------
hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
if not hf_token:
    raise RuntimeError("Set HF_TOKEN in environment, e.g. export HF_TOKEN='hf_xxx'")

login(token=hf_token)

# ------- Load tokenizer -------
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# Ensure pad token exists
added_tokens = False
if tokenizer.pad_token_id is None:
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    added_tokens = True

# ------- Load base model -------
model = None
try:
    from transformers import Gemma3ForCausalLM  # type: ignore
    print("Loading model using Gemma3ForCausalLM.from_pretrained(...)")
    model = Gemma3ForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
        low_cpu_mem_usage=True,
    )
except Exception as e:
    print("Gemma3ForCausalLM not available or failed to load (falling back). Exception:", e)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
        low_cpu_mem_usage=True,
    )

# ------- LoRA config -------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

# If we added tokens to tokenizer, resize model embeddings before PEFT wrapping
if added_tokens:
    model.resize_token_embeddings(len(tokenizer))

# ------- Load existing adapter weights as TRAINABLE -------
adapter_loaded = False
if load_existing_adapter and os.path.isdir(output_adapter_dir):
    try:
        print(f"Found existing adapter directory at {output_adapter_dir!r}, loading it...")
        model = PeftModel.from_pretrained(
            model,
            output_adapter_dir,
            device_map="auto",
            is_trainable=True,   # IMPORTANT: makes adapter weights require gradients
        )
        adapter_loaded = True
        print("Successfully loaded existing LoRA adapter; continuing training.")
    except Exception as e:
        print("Warning: failed to load existing adapter; training will start fresh. Exception:", e)
        adapter_loaded = False
else:
    print(f"No existing adapter found at {output_adapter_dir!r}; starting a new LoRA training run.")

# Ensure caches off for training
model.config.use_cache = False

# Optional sanity check
def print_trainable_params(m):
    trainable = 0
    total = 0
    for p in m.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    print(f"Trainable params: {trainable} / {total}")

print_trainable_params(model)

# ------- Load training data -------
if not os.path.isfile(facts_path):
    raise RuntimeError(f"Training data not found at {facts_path!r}. Current working directory: {os.getcwd()}")

with open(facts_path, "r", encoding="utf-8") as f:
    examples = json.load(f)

if not isinstance(examples, list):
    raise RuntimeError(f"Expected a list of examples in {facts_path!r}.")

train_ds = Dataset.from_list(examples)

# ------- Format examples to a single 'text' field -------
def format_example(ex):
    system_msg = "Sa oled abivalmis assistent. Vasta lühidalt ja eesti keeles."
    user_msg = ex.get("question", "")
    assistant_msg = ex.get("answer", "")

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,  # important for supervised fine-tuning
            )
            return {"text": text}
        except Exception:
            pass

    # Fallback plain text format
    text = f"SYSTEM: {system_msg}\n\nUSER: {user_msg}\n\nASSISTANT: {assistant_msg}"
    return {"text": text}

train_ds = train_ds.map(format_example)

if "text" not in train_ds.column_names:
    raise RuntimeError("After mapping, train_ds does not contain a 'text' column. Check format_example().")

# ------- TrainingArguments -------
args = TrainingArguments(
    output_dir=checkpoint_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    learning_rate=5e-5,
    num_train_epochs=3,
    logging_steps=1,
    save_steps=50,
    bf16=(torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)()),
    report_to="none",
)

# ------- SFTTrainer -------
trainer_kwargs = {
    "model": model,
    "train_dataset": train_ds,
    "args": args,
    "processing_class": tokenizer,
}

# Only pass peft_config if we are training from the base model.
# If adapter is already loaded, do NOT re-wrap.
if not adapter_loaded:
    trainer_kwargs["peft_config"] = lora_config
else:
    print("Adapter already loaded: not passing peft_config to avoid re-wrapping.")

# Filter kwargs by SFTTrainer signature for compatibility across TRL versions
sig = inspect.signature(SFTTrainer.__init__)
accepted = set(sig.parameters.keys())
accepted.discard("self")
filtered_kwargs = {k: v for k, v in trainer_kwargs.items() if k in accepted}

print("SFTTrainer supported parameters:", sorted(accepted))
print("Passing parameters:", sorted(filtered_kwargs.keys()))

try:
    trainer = SFTTrainer(**filtered_kwargs)
except TypeError as te:
    print("SFTTrainer construction failed with TypeError:", te)
    print("Retrying minimal constructor...")
    if adapter_loaded:
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_ds,
            args=args,
            processing_class=tokenizer,
        )
    else:
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_ds,
            args=args,
            peft_config=lora_config,
            processing_class=tokenizer,
        )

# ------- Train -------
# IMPORTANT: do NOT resume from checkpoint here when you already loaded adapter weights manually.
trainer.train()

# ------- Save adapter and tokenizer -------
try:
    trainer.model.save_pretrained(output_adapter_dir)
except Exception:
    model.save_pretrained(output_adapter_dir)

tokenizer.save_pretrained(output_adapter_dir)

print(f"Done. Saved LoRA adapter and tokenizer to: {output_adapter_dir}")
