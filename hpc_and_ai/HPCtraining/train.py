# training_fixed.py
# Requirements (run before):
# pip install -U pip
# pip install -U torch transformers accelerate peft trl datasets bitsandbytes huggingface_hub

import os
import json
import inspect
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer
from huggingface_hub import login

# ------- Config -------
model_id = "google/gemma-3-1b-it"
facts_path = "huvitavad_faktid.json"   # adjust path if your file is named differently
output_adapter_dir = "gemma_1.0_lora"
checkpoint_dir = "checkpoint_lora"

# ------- HF login -------
hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
if not hf_token:
    raise RuntimeError("Set HF_TOKEN in environment, e.g. export HF_TOKEN='hf_xxx'")

login(token=hf_token)

# ------- Load tokenizer -------
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# ensure pad token exists
added_tokens = False
if tokenizer.pad_token_id is None:
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    added_tokens = True

# ------- Load model: try explicit Gemma causal class, fallback to AutoModelForCausalLM -------
model = None
try:
    # Try explicit causal Gemma class (if available)
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

# ensure caches off for training
model.config.use_cache = False

# If you plan QLoRA (k-bit) prepare model here:
# model = prepare_model_for_kbit_training(model)

# ------- PEFT LoRA config (do NOT call get_peft_model here) -------
# We'll pass this config to SFTTrainer which will apply the adapter itself.
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

# If we added tokens to tokenizer, resize model embeddings BEFORE trainer wraps it with PEFT
if added_tokens:
    model.resize_token_embeddings(len(tokenizer))

# ------- Load training data -------
if not os.path.isfile(facts_path):
    raise RuntimeError(f"Training data not found at {facts_path!r}. Current working directory: {os.getcwd()}")

with open(facts_path, "r", encoding="utf-8") as f:
    examples = json.load(f)

if not isinstance(examples, list):
    raise RuntimeError(f"Expected a list of examples in {facts_path!r}.")

train_ds = Dataset.from_list(examples)

# ------- Format examples to single 'text' field -------
def format_example(ex):
    system_msg = "Sa oled abivalmis assistent. Vasta lühidalt ja eesti keeles."
    user_msg = ex.get("question", "")
    assistant_msg = ex.get("answer", "")

    # Prefer tokenizer helper if present — safe fallback if it errors
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
                add_generation_prompt=False
            )
            return {"text": text}
        except Exception:
            pass

    # Fallback: a simple readable chat-like text
    text = f"SYSTEM: {system_msg}\n\nUSER: {user_msg}\n\nASSISTANT: {assistant_msg}"
    return {"text": text}

train_ds = train_ds.map(format_example)

if "text" not in train_ds.column_names:
    raise RuntimeError("After mapping, train_ds does not contain a 'text' column. Check format_example.")

# ------- TrainingArguments -------
args = TrainingArguments(
    output_dir=checkpoint_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    num_train_epochs=1,
    logging_steps=1,
    save_steps=50,
    bf16=(torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)()),
    report_to="none",
)

# ------- SFTTrainer -------
# Build trainer kwargs; pass the base model and peft_config (trainer will apply PEFT)
trainer_kwargs = {
    "model": model,               # plain base model, NOT wrapped with get_peft_model
    "train_dataset": train_ds,
    "peft_config": lora_config,   # trainer will internally call get_peft_model
    "args": args,
    # pass tokenizer as processing_class to avoid AutoProcessor loading for multimodal configs
    "processing_class": tokenizer,
}

# Filter kwargs by signature so this script works across TRL versions
sig = inspect.signature(SFTTrainer.__init__)
accepted = set(sig.parameters.keys())
accepted.discard("self")
filtered_kwargs = {k: v for k, v in trainer_kwargs.items() if k in accepted}

print("SFTTrainer supported parameters:", sorted(accepted))
print("Passing parameters:", sorted(filtered_kwargs.keys()))

try:
    trainer = SFTTrainer(**filtered_kwargs)
except TypeError as te:
    # fallback to a minimal constructor; trainer will still receive peft_config in later call if supported
    print("SFTTrainer construction failed with TypeError:", te)
    print("Retrying minimal constructor (model, train_dataset, args, peft_config)...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        args=args,
        peft_config=lora_config,
    )

# ------- Train -------
trainer.train()

# ------- Save adapter and tokenizer -------
# After trainer wraps the base model with PEFT, trainer.model is the PeftModel.
# Save the adapter (LoRA) + tokenizer.
try:
    # trainer.model is typically the PeftModel after trainer setup
    trainer.model.save_pretrained(output_adapter_dir)
except Exception:
    # Fallback: if trainer didn't wrap and model is still base, try saving via model.save_pretrained
    model.save_pretrained(output_adapter_dir)

tokenizer.save_pretrained(output_adapter_dir)

print(f"Done. Saved LoRA adapter and tokenizer to: {output_adapter_dir}")
