from transformers import AutoProcessor, AutoModelForCausalLM
import torch

# Lae processor ja mudel
processor = AutoProcessor.from_pretrained("google/gemma-3-270m", device_map="auto")
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-270m", device_map="auto")

prompt = (
    "You are a polite, informative assistant who speaks only English. "
    "Answer only the question that has been asked. "
    "Answer in 1-2 full sentences. "
    "Do not repeat yourself."
    "\nUser: how old are you"
    "\nAssistant:"
)

# Tokenize
inputs = processor(prompt, return_tensors="pt").to(model.device)

# Generate
out = model.generate(
    **inputs,
    max_new_tokens=32,
    pad_token_id=processor.eos_token_id
)

# Decode
text = processor.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

# Võta ainult esimene lause
first_answer = text.split(".")[0].strip() + "."
print(first_answer)
