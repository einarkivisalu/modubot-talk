# train adapter base with analysis (plotting)
# nb! original dataset gets split into train and test

import os
import json
import inspect
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer
from huggingface_hub import login
import matplotlib.pyplot as plt

MODEL_ID = "google/gemma-3-1b-it"


def train_domain_adapter(
    model_id: str,
    data_path: str,
    output_adapter_dir: str,
    checkpoint_dir: str,
    domain_name: str,
    system_msg: str,
    num_train_epochs: int = 3,
    learning_rate: float = 5e-5,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 2,
    max_seq_length: int = 1024,
    eval_split_ratio: float = 0.1, #for analysis train and testdata split (right now 90% training, 10% evaluation)

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
        # specific loader for gemma
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
        # generic loader
        # load whatever model this is as casual text generator (predict one word at a time)
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

    dataset = Dataset.from_list(examples)

    # split dataset into training and testing
    if len(dataset) < 2:
        raise RuntimeError("Need at least 2 examples to create a train/eval split.")

    # random split, train_test_split() [huggingface] shuffles the dataset before splitting by default
    split = dataset.train_test_split(test_size=eval_split_ratio, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]

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
                    add_generation_prompt=False,
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
    eval_ds = eval_ds.map(format_example)

    if "text" not in train_ds.column_names:
        raise RuntimeError("After mapping, train_ds does not contain a 'text' column.")
    if "text" not in eval_ds.column_names:
        raise RuntimeError("After mapping, eval_ds does not contain a 'text' column.")

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
        max_grad_norm=1.0,
        bf16=False,
        fp16=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to="none",
        optim="adamw_torch",
    )

    trainer_kwargs = {
        "model": model,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
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

    trainer.save_state()

    #-----------------------------------------------------------------------
    # plot for analysis
    os.makedirs(trainer.args.output_dir, exist_ok=True)
    log_path = os.path.join(trainer.args.output_dir, "log_history.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=2)

    print(f"Saved log history to {log_path}")

    with open(log_path, "r", encoding="utf-8") as f:
        log_history = json.load(f)

    train_losses = [log["loss"] for log in log_history if "loss" in log]
    epoch_train = [log.get("epoch") for log in log_history if "loss" in log]

    eval_losses = [log["eval_loss"] for log in log_history if "eval_loss" in log]
    epoch_eval = [log.get("epoch") for log in log_history if "eval_loss" in log]

    plt.figure(figsize=(8, 5))
    plt.plot(epoch_train, train_losses, label="Training Loss")
    if eval_losses:
        plt.plot(epoch_eval, eval_losses, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss per Epoch")
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(trainer.args.output_dir, "loss_curve.png")
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {plot_path}")
    #----------------------------------------------------------------------------

    print("Training finished.")

    os.makedirs(output_adapter_dir, exist_ok=True)
    try:
        trainer.model.save_pretrained(output_adapter_dir)
    except Exception:
        model.save_pretrained(output_adapter_dir)

    tokenizer.save_pretrained(output_adapter_dir)
    print(f"[{domain_name}] Done. Saved LoRA adapter and tokenizer to: {output_adapter_dir}")


if __name__ == "__main__":
    print("treening vajab parameetreid - treeni teise faili kaudu mis sisaldab parameetreid. nt train_facts_adapter.py.")
