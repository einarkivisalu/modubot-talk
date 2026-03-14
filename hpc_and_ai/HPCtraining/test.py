# hpc_test_fix_gemma.py
"""
HPC test loader that forces Gemma causal class, supports safetensors adapters, prints diagnostics.
"""
import os
import sys
from pathlib import Path
import torch

# config
BASE_MODEL = "google/gemma-3-1b-it"
ADAPTER_PATH = Path("./gemma_1.0_lora")
PROMPT_USER = "Räägi mulle üks huvitav fakt."
MAX_NEW_TOKENS = 120
USE_8BIT_FALLBACK = True

print("Python:", sys.version.splitlines()[0])
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Target device:", device)
print("Adapter path exists:", ADAPTER_PATH.exists())
print("Adapter contents:", sorted(p.name for p in ADAPTER_PATH.iterdir()) if ADAPTER_PATH.exists() else "N/A")

# --- tokenizer (prefer adapter tokenizer) ---
from transformers import AutoTokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER_PATH), trust_remote_code=True)
    print("Loaded tokenizer from adapter folder.")
except Exception as e:
    print("Adapter tokenizer load failed:", e)
    print("Falling back to base tokenizer.")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

# ensure pad token
if tokenizer.pad_token_id is None:
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    print("Added pad token to tokenizer.")
tokenizer.model_max_length = getattr(tokenizer, "model_max_length", 2048)

# --- load base model: force Gemma causal class if available ---
from transformers import AutoModelForCausalLM
base_model = None
try:
    # try explicit Gemma causal class first (preferred)
    from transformers import Gemma3ForCausalLM  # type: ignore
    print("Trying explicit Gemma3ForCausalLM.from_pretrained(...)")
    base_model = Gemma3ForCausalLM.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
        ignore_mismatched_sizes=True,
    )
except Exception as e:
    print("Explicit Gemma3ForCausalLM not available or failed:", e)
    # try normal AutoModelForCausalLM (but still set device_map)
    def try_load(load_in_8bit=False):
        kwargs = dict(trust_remote_code=True, low_cpu_mem_usage=True)
        kwargs["device_map"] = "auto" if device == "cuda" else None
        if load_in_8bit:
            # require bitsandbytes installed
            kwargs["load_in_8bit"] = True
            print("Loading base model in 8-bit mode.")
        print("Loading base with kwargs:", {k:v for k,v in kwargs.items() if k!="trust_remote_code"})
        return AutoModelForCausalLM.from_pretrained(BASE_MODEL, **kwargs)

    try:
        base_model = try_load(load_in_8bit=False)
    except Exception as e_full:
        print("Full load failed:", e_full)
        if USE_8BIT_FALLBACK:
            try:
                base_model = try_load(load_in_8bit=True)
            except Exception as e8:
                print("8-bit load also failed:", e8)
                raise SystemExit("Could not load base model.")
        else:
            raise SystemExit("Could not load base model and 8-bit disabled.")

print("Base model loaded. Model class:", base_model.__class__.__name__)

# --- wrap with adapter using PEFT, detect safetensors file if present ---
from peft import PeftModel

use_safetensors = False
if (ADAPTER_PATH / "adapter_model.safetensors").exists():
    use_safetensors = True
    print("Detected adapter_model.safetensors -> will pass use_safetensors=True to PeftModel.from_pretrained()")

print("Applying LoRA adapter from:", ADAPTER_PATH)
try:
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_PATH),
                                      device_map="auto" if device == "cuda" else None,
                                      torch_dtype=getattr(torch, "float16") if device=="cuda" else None,
                                      use_safetensors=use_safetensors)
    print("PeftModel.from_pretrained succeeded (with device_map).")
except Exception as e1:
    print("PeftModel.from_pretrained with device_map failed:", e1)
    print("Retrying PeftModel.from_pretrained without device_map and without torch_dtype...")
    try:
        model = PeftModel.from_pretrained(base_model, str(ADAPTER_PATH), use_safetensors=use_safetensors)
        print("PeftModel.from_pretrained succeeded on retry.")
    except Exception as e2:
        print("PeftModel.from_pretrained final attempt failed:", e2)
        raise SystemExit("Failed to load LoRA adapter onto base model.")

# --- diagnostic: compare adapter keys vs model state dict keys (short summary) ---
try:
    import peft
    # attempt to read keys inside safetensors or pytorch checkpoint
    ckpt_file = ADAPTER_PATH / "adapter_model.safetensors"
    if not ckpt_file.exists():
        ckpt_file = ADAPTER_PATH / "pytorch_model.bin"
    if ckpt_file.exists():
        print("Adapter checkpoint file:", ckpt_file.name)
    else:
        print("No single adapter checkpoint file found (adapter saved as multiple files).")
except Exception:
    pass

# Print warning if PEFT reported missing adapter keys earlier (user gets warning in logs)
print("Model class after PEFT wrapping:", model.__class__.__name__)
# check and print device
param = next(model.parameters())
print("Model parameters device:", param.device)

# ensure embeddings resize if needed
try:
    embeddings = model.get_input_embeddings()
    if embeddings is not None and embeddings.num_embeddings != len(tokenizer):
        print("Resizing token embeddings from", embeddings.num_embeddings, "to", len(tokenizer))
        model.resize_token_embeddings(len(tokenizer))
except Exception as e:
    print("Embedding resize check failed:", e)

model.eval()

# --- build prompt: try apply_chat_template with add_generation_prompt=True first, fallback if fails ---
system = "Sa oled abivalmis assistent. Vasta lühidalt ja eesti keeles."
try:
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role":"system","content":system},{"role":"user","content":PROMPT_USER}]
        # if training used add_generation_prompt=False for formatting, for generation we want add_generation_prompt=True
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        print("Using tokenizer.apply_chat_template(add_generation_prompt=True).")
    else:
        prompt = f"SYSTEM: {system}\n\nUSER: {PROMPT_USER}\n\nASSISTANT:"
except Exception as e:
    print("apply_chat_template failed, fallback. Error:", e)
    prompt = f"SYSTEM: {system}\n\nUSER: {PROMPT_USER}\n\nASSISTANT:"

print("\nPrompt:\n", prompt)

# --- tokenize & generate ---
inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(next(model.parameters()).device)

gen_kwargs = dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=False, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
print("\nGenerating...")
with torch.no_grad():
    out_ids = model.generate(**inputs, **gen_kwargs)

decoded = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
print("\n=== RESPONSE ===\n", decoded[0])
