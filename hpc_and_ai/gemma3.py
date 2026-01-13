import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen3-1.7B"

# Soovi korral vähenda logimüra
# os.environ["TRANSFORMERS_VERBOSITY"] = "error"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16
)

def strip_think(text: str) -> str:
    # Eemalda <think>...</think> (kui ilmub)
    if "<think>" in text and "</think>" in text:
        text = text.split("</think>", 1)[1]
    # Eemalda üksikud <think> markerid, kui tag end puudu
    text = text.replace("<think>", "").replace("</think>", "")
    return text.strip()

def looks_english(text: str) -> bool:
    """
    Heuristika: kui tekstis on palju levinud inglise sõnu ja väga vähe eesti täpitähti,
    käsitleme seda inglise keele tõenäosusena.
    """
    t = text.lower()
    common_en = ["okay", "the user", "let me", "i know", "which i", "but they", "in english", "from my knowledge"]
    en_hits = sum(1 for p in common_en if p in t)

    # eesti täpitähed (väike signaal; võib puududa ka eesti tekstis, aga aitab)
    et_chars = sum(t.count(c) for c in "äöüõ")

    # Kui “meta inglise” fraase on >=1 ja eesti täpitähti pole üldse, siis tõenäoliselt vale keel
    return (en_hits >= 1 and et_chars == 0)

def generate_estonian(user_question: str) -> str:
    system = (
        "REEGLID:\n"
        "1) Vasta AINULT eesti keeles.\n"
        "2) ÄRA kirjuta mõttekäiku, analüüsi, vahepealseid samme ega meta-juttu.\n"
        "3) ÄRA kasuta inglise keelt.\n"
        "4) Kui sul puudub info, ütle: 'Mul puudub selle kohta info.'\n"
        "5) Kirjuta lühidalt ja selgelt.\n\n"
        "/no_think\n"
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_question}
    ]

    # Qwen3: thinking tuleb välja lülitada SIIN (chat template’is)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=160,
            do_sample=False,
            repetition_penalty=1.10,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    text = strip_think(text)

    # Fallback: kui tuli ikka inglise/meta, siis sunni ümberkirjutus eesti keelde
    if looks_english(text):
        messages2 = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    "Kirjuta allolev vastus ümber AINULT eesti keelde, selgelt ja faktipõhiselt. "
                    "Ära lisa kommentaare ega meta-juttu.\n\n"
                    f"Vastus:\n{text}"
                )
            }
        ]
        prompt2 = tokenizer.apply_chat_template(
            messages2,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        inputs2 = tokenizer(prompt2, return_tensors="pt").to(model.device)

        with torch.inference_mode():
            out2 = model.generate(
                **inputs2,
                max_new_tokens=200,
                do_sample=False,
                repetition_penalty=1.10,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )

        text2 = tokenizer.decode(out2[0][inputs2["input_ids"].shape[1]:], skip_special_tokens=True)
        text = strip_think(text2)

    return text.strip()

if __name__ == "__main__":
    print(generate_estonian("Mis on tehisnärvivõrk?"))
