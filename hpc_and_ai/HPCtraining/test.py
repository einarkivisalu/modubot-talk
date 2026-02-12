# -----------------------------
# 7) Test (in the same script)
# -----------------------------
def chat(prompt: str, max_new_tokens: int = 50):
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer briefly."},
        {"role": "user", "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        out = trainer.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    # Decode only the new tokens
    new_tokens = out[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

print(chat("Why is the sky blue?"))