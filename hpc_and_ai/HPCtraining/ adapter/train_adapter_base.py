# pip install -U torch transformers accelerate peft trl datasets bitsandbytes huggingface_hub
# pip install -U sentence-transformers scikit-learn joblib

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
    data_path: str = "",
    output_adapter_dir: str = "",
    checkpoint_dir: str = "",
    domain_name: str = "",
    system_msg: str = "",
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
            device_map="auto",
            trust_remote_code=True,
            ignore_mismatched_sizes=True,
            low_cpu_mem_usage=True,
        )
    except Exception as e:
        print("Gemma3ForCausalLM failed, falling back to AutoModelForCausalLM. Exception:", e)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            trust_remote_code=True,
            ignore_mismatched_sizes=True,
            low_cpu_mem_usage=True,
        )

    model.config.use_cache = False

    if added_tokens:
        model.resize_token_embeddings(len(tokenizer))

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
        bf16=(torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)()),
        report_to="none",
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

    try:
        trainer = SFTTrainer(**filtered_kwargs)
    except TypeError as te:
        print("SFTTrainer construction failed with TypeError:", te)
        print("Retrying minimal constructor...")
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_ds,
            args=args,
            peft_config=lora_config,
            processing_class=tokenizer,
        )

    trainer.train()

    os.makedirs(output_adapter_dir, exist_ok=True)
    try:
        trainer.model.save_pretrained(output_adapter_dir)
    except Exception:
        model.save_pretrained(output_adapter_dir)

    tokenizer.save_pretrained(output_adapter_dir)
    print(f"[{domain_name}] Done. Saved adapter and tokenizer to: {output_adapter_dir}")
