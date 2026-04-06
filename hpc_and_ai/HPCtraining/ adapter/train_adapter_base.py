# training_gpu.py

import os
import json
import inspect
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer
from huggingface_hub import login

MODEL_ID = "google/gemma-3-1b-it"


def train_domain_adapter(
    model_id: str = MODEL_ID,
    data_path: str = "/gpfs/mariana/home/anemoo/training_materials/huvitavad_faktid.json",
    output_adapter_dir: str = "/gpfs/mariana/home/anemoo/adapter_training/adapters",
    checkpoint_dir: str = "/gpfs/mariana/home/anemoo/adapter_training/checkpoints",
    domain_name: str = "faktid",
    system_msg: str = "Sa oled abivalmis assistent. Vasta lühidalt ja eesti keeles.",
    num_train_epochs: int = 2,
    learning_rate: float = 5e-5,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 2,
    max_seq_length: int = 1024,
):
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not hf_token:
        raise RuntimeError("Set HF_TOKEN in environment, e.g. export HF_TOKEN='hf_xxx'")

    login(token=hf_token)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in this environment. "
            "Install a CUDA-compatible PyTorch wheel for your driver, then rerun."
        )

    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    print("CUDA device name:", torch.cuda.get_device_name(0))

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    added_tokens = False
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
        added_tokens = True

    model = None
    try:
        from transformers import Gemma3ForCausalLM  # type: ignore
        print("Loading model using Gemma3ForCausalLM.from_pretrained(...)")
        model = Gemma3ForCausalLM.from_pretrained(
            model_id,
            device_map=None,
            trust_remote_code=True,
            ignore_mismatched_sizes=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
    except Exception as e:
        print("Gemma3ForCausalLM failed, falling back to AutoModelForCausalLM. Exception:", e)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=None,
            trust_remote_code=True,
            ignore_mismatched_sizes=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )

    model.config.use_cache = False
    model = model.to("cuda")

    if added_tokens:
        model.resize_token_embeddings(len(tokenizer))

    print("Model loaded on:", next(model.parameters()).device)

    if not os.path.isfile(data_path):
        raise RuntimeError(
            f"Training data not found at {data_path!r}. Current working directory: {os.getcwd()}"
        )

    with open(data_path, "r", encoding="utf-8") as f:
        examples = json.load(f)

    if not isinstance(examples, list):
        raise RuntimeError(f"Expected a list of examples in {data_path!r}.")

    train_ds = Dataset.from_list(examples)

    def format_example(ex):
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
                    add_generation_prompt=True,
                )
                return {"text": text}
            except Exception as e:
                print("apply_chat_template failed:", e)

        text = (
            f"SYSTEM: {system_msg}\n\n"
            f"USER: {user_msg}\n\n"
            f"ASSISTANT: {assistant_msg}"
        )
        return {"text": text}

    train_ds = train_ds.map(format_example)

    if "text" not in train_ds.column_names:
        raise RuntimeError("After mapping, train_ds does not contain a 'text' column.")

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

    args = TrainingArguments(
        output_dir=checkpoint_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        logging_steps=1,
        save_steps=50,
        max_grad_norm=1.0,
        bf16=False,
        fp16=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to="none",
        optim="adamw_torch",
    )

    trainer_kwargs = {
        "model": model,
        "train_dataset": train_ds,
        "peft_config": lora_config,
        "args": args,
        "processing_class": tokenizer,
        "dataset_text_field": "text",
        "max_seq_length": max_seq_length,
    }

    sig = inspect.signature(SFTTrainer.__init__)
    accepted = set(sig.parameters.keys())
    accepted.discard("self")
    filtered_kwargs = {k: v for k, v in trainer_kwargs.items() if k in accepted}

    print("SFTTrainer supported parameters:", sorted(accepted))
    print("Passing parameters:", sorted(filtered_kwargs.keys()))

    trainer = SFTTrainer(**filtered_kwargs)

    print("Starting training on GPU...")
    trainer.train()
    print("Training finished.")

    os.makedirs(output_adapter_dir, exist_ok=True)
    try:
        trainer.model.save_pretrained(output_adapter_dir)
    except Exception:
        model.save_pretrained(output_adapter_dir)

    tokenizer.save_pretrained(output_adapter_dir)
    print(f"[{domain_name}] Done. Saved LoRA adapter and tokenizer to: {output_adapter_dir}")


if __name__ == "__main__":
    train_domain_adapter()
