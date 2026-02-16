import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ---- SETTINGS ----
base_model = "google/gemma-3-1b-it"
adapter_path = "gemma_1.0"
# ------------------

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)

# Load LoRA adapter
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

# QUESTION
question = "Mis on Eesti pealinn?"

# same chat template as training
messages = [
    {"role": "system", "content": "Sa oled abivalmis assistent. Vasta lühidalt ja eesti keeles."},
    {"role": "user", "content": question},
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True  # important for inference
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
    )

print(tokenizer.decode(output[0], skip_special_tokens=True))
