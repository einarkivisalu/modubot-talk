import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ----------------------
# Model
# ----------------------
MODEL_ID = "google/gemma-3-1b-it"   # instruction-tuned (important)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Force full CPU load (NO auto sharding, NO disk offload)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map={"": "cpu"},
    torch_dtype=torch.float32
)

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
text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

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

answer = " ".join(sentences) if sentences else text.split("\n")[0]
print(answer)
