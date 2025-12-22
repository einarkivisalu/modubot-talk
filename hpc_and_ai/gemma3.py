from transformers import AutoProcessor, AutoModelForCausalLM
import torch

# Load processor and model
processor = AutoProcessor.from_pretrained("google/gemma-3-270m", device_map="auto")
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-270m", device_map="auto")

# System + user prompt
prompt = (
    "You are a polite, informative assistant who speaks only English. "
    "Answer only the question that has been asked. "
    "Answer in 1-2 full sentences. "
    "Do not repeat yourself."
    "\nUser: What is the best technique for knitting?"
    "\nAssistant:"
)

# Tokenize prompt
inputs = processor(prompt, return_tensors="pt").to(model.device)

# Generate output (deterministic)
out = model.generate(
    **inputs,
    max_new_tokens=64,
    pad_token_id=processor.eos_token_id,
    do_sample=False  # deterministic, prevents ping-pong
)

# Decode output
text = processor.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

# Take only the first sentence
first_answer = text.split(".")[0].strip() + "."
print(first_answer)
