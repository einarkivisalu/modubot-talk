import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ----------------------
# Model
# ----------------------
MODEL_ID = "google/gemma-3-4b-it"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)


DTYPE = torch.float32

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    low_cpu_mem_usage=True
).eval()

# ----------------------
# Estonian prompt
# ----------------------
messages = [
    {
        "role": "user",
        "content": (
            "Vasta ainult eesti keeles. "
            "Vasta 1–2 täislausega. Ära korda ennast.\n\n"
            "Mis on tehisnärvivõrk?"
        )
    }
]

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
# Decode (1–2 lauset)
# ----------------------
text = tokenizer.decode(
    out[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens=True
).strip()

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
