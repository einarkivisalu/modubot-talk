# hpc_test_fix_gemma_minimal.py
"""
Minimal CPU-only test: loads base + LoRA adapter and prints only the generated text.
Errors and diagnostics go to stderr; stdout contains exactly the answer.
"""
import sys
from pathlib import Path
import torch

# --- config (edit if necessary) ---
BASE_MODEL = "google/gemma-3-1b-it"
ADAPTER_PATH = Path("EDITPATH\\gemma_1.0_lora")
PROMPT_USER = "Räägi mulle üks huvitav fakt."
MAX_NEW_TOKENS = 120

# reduce HF logging noise (warnings/errors still go to stderr)
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

# load tokenizer (prefer adapter tokenizer)
from transformers import AutoTokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER_PATH), trust_remote_code=True)
except Exception:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

# ensure pad token
if tokenizer.pad_token_id is None:
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

# force CPU mode
device = torch.device("cpu")

# load base model (try Gemma class, otherwise AutoModelForCausalLM)
from transformers import AutoModelForCausalLM
base_model = None
try:
    from transformers import Gemma3ForCausalLM  # type: ignore
    base_model = Gemma3ForCausalLM.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map=None,  # keep on CPU
    )
except Exception:
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map=None,
    )

# apply LoRA adapter using PEFT (detect safetensors)
from peft import PeftModel
use_safetensors = (ADAPTER_PATH / "adapter_model.safetensors").exists()
try:
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_PATH), use_safetensors=use_safetensors, device_map=None)
except Exception as e:
    # try minimal args (still CPU)
    try:
        model = PeftModel.from_pretrained(base_model, str(ADAPTER_PATH), use_safetensors=use_safetensors)
    except Exception as e2:
        print("ERROR: failed to load adapter: " + str(e2), file=sys.stderr)
        raise SystemExit(1)

# resize embeddings if tokenizer changed vocab size
try:
    embeddings = model.get_input_embeddings()
    if embeddings is not None and embeddings.num_embeddings != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
except Exception:
    pass

model.eval()

# build prompt (try chat template)
system = "Sa oled abivalmis assistent. Vasta lühidalt ja eesti keeles."
try:
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role":"system","content":system},{"role":"user","content":PROMPT_USER}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = f"SYSTEM: {system}\n\nUSER: {PROMPT_USER}\n\nASSISTANT:"
except Exception:
    prompt = f"SYSTEM: {system}\n\nUSER: {PROMPT_USER}\n\nASSISTANT:"

# tokenize -> put tensors on CPU explicitly
inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
inputs = {k: v.to(device) for k, v in inputs.items()}

gen_kwargs = dict(
    max_new_tokens=MAX_NEW_TOKENS,
    do_sample=False,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)

# generate and print only the decoded text (no extra formatting)
with torch.no_grad():
    out_ids = model.generate(**inputs, **gen_kwargs)

decoded = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
# print only the first output, stripped of surrounding whitespace/newlines
print(decoded[0].strip())
