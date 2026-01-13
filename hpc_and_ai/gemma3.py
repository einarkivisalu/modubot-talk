import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ----------------------
# Model
# ----------------------
MODEL_ID = "google/gemma-3-4b-it"  # instruction-tuned

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Option A: reduce RAM by using bf16 (preferred) or fp16 on CPU + low peak RAM loading
# bfloat16 works on many CPUs; if your CPU/PyTorch build doesn't support it, the code falls back to float16.
preferred_dtype = torch.bfloat16
fallback_dtype = torch.float16

def load_model(dtype: torch.dtype):
    return AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map={"": "cpu"},        # force CPU
        dtype=dtype,                   # use dtype (torch_dtype deprecation)
        low_cpu_mem_usage=True         # reduces peak RAM while loading shards
    )

try:
    model = load_model(preferred_dtype)
except Exception as e:
    print(f"bf16 load failed ({type(e).__name__}: {e})")
    print("Falling back to float16...")
    model = load_model(fallback_dtype)

model.eval()

# ----------------------
# Estonian prompt
# ----------------------
messages = [
    {
        "role": "user",
        "content": (
            "Vasta ainult eesti keeles. "
            "Vasta 1–2 täislausega. Ära korda ennast.\n\n"
            "Kuidas mu lastelastel läheb?"
        )
    }
]

# Use Gemma chat formatting
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# ----------------------
# Tokenize
# ----------------------
inputs = tokenizer(prompt, return_tensors="pt")

# ----------------------
# Generate
# ----------------------
with torch.inference_mode():
    out = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=False,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )

# ----------------------
# Decode
# ----------------------
text = tokenizer.decode(
    out[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens=True
).strip()

# Keep 1–2 sentences
sentences = []
buf = ""
for ch in text:
    buf += ch
    if ch in ".!?":
        sentences.append(buf.strip())
        buf = ""
    if len(sentences) == 2:
        break

answer = " ".join(sentences) if sentences else text.split("\n")[0].strip()
print(answer)
