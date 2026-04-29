from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import requests
import torch
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import Gemma3ForCausalLM
except Exception:
    Gemma3ForCausalLM = None


# =============================================================================
# CONFIG
# =============================================================================

MODEL_ID = "google/gemma-3-1b-it"

ROUTER_PATHS = [
    "router_artifacts/router_context.joblib",
    "router_artifacts/router_baseline.joblib",
]

ADAPTERS = {
    "facts": "adapters1.1/facts",
    "kasitoo": "adapters1.1/kasitoo",
    "luuletused": "adapters1.1/luuletused",
    "muistendid": "adapters1.1/muistendid",
    "tahtpaevad": "adapters1.1/tahtpaevad",
}

SYSTEM_MSGS = {
    "facts": "Sa oled abivalmis assistent. Vasta lühidalt ja eesti keeles.",
    "kasitoo": "Sa oled abivalmis assistent. Vasta lühidalt ja eesti keeles. Teema on käsitöö.",
    "luuletused": "Sa oled abivalmis assistent. Vasta lühidalt ja eesti keeles. Teema on luuletused.",
    "tahtpaevad": "Sa oled abivalmis assistent. Vasta lühidalt ja eesti keeles. Teema on tähtpäevad.",
    "muistendid": "Sa oled abivalmis assistent. Vasta lühidalt ja eesti keeles. Teema on muistendid.",
}

STATE_PATH = Path("conversation_state.json")


# =============================================================================
# STATE
# =============================================================================

@dataclass
class State:
    last_topic: str = "unknown"
    turn_index: int = 1

    def save(self):
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f)

    @classmethod
    def load(cls):
        if STATE_PATH.exists():
            try:
                data = json.load(open(STATE_PATH))
                return cls(
                    last_topic=data.get("last_topic", "unknown"),
                    turn_index=int(data.get("turn_index", 1)),
                )
            except:
                pass
        return cls()


# =============================================================================
# ROUTER
# =============================================================================

class Router:
    def __init__(self, path):
        data = joblib.load(path)
        self.model = data["classifier"]
        self.embedder = SentenceTransformer(data["embedding_model_name"])

    def predict(self, text):
        emb = self.embedder.encode([text])
        probs = self.model.predict_proba(emb)[0]
        idx = np.argmax(probs)
        return self.model.classes_[idx], float(probs[idx])


def load_router():
    for p in ROUTER_PATHS:
        if Path(p).exists():
            return Router(p)
    raise FileNotFoundError("Router puudub")


# =============================================================================
# MODEL + ADAPTERS
# =============================================================================

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if Gemma3ForCausalLM:
        model = Gemma3ForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    return tokenizer, model


def load_adapters(model):
    first = True
    for name, path in ADAPTERS.items():
        if not Path(path).exists():
            continue

        if first:
            model = PeftModel.from_pretrained(model, path, adapter_name=name)
            first = False
        else:
            model.load_adapter(path, adapter_name=name)

    return model


# =============================================================================
# SEARCH (IMPROVED)
# =============================================================================

def should_search(question: str, topic: str, confidence: float) -> bool:
    q = question.lower()

    strong = [
        "ilm", "täna", "praegu", "mis kell",
        "uudised", "internet", "kust", "kus"
    ]

    if any(s in q for s in strong):
        return True

    if confidence < 0.55:
        return True

    if topic in ["luuletused", "muistendid"] and any(s in q for s in strong):
        return True

    return False


def search(query):
    try:
        r = requests.get(
            "http://localhost:8080/search",
            params={"q": query, "format": "json"},
            timeout=10,
        )
        data = r.json()

        results = []
        for item in data.get("results", [])[:5]:
            results.append(f"{item.get('title','')}\n{item.get('content','')}")

        return "\n\n".join(results)
    except Exception as e:
        return f"Otsing ebaõnnestus: {e}"


# =============================================================================
# GENERATION
# =============================================================================

def build_prompt(tokenizer, question, topic, search_ctx=None):
    system = SYSTEM_MSGS.get(topic, SYSTEM_MSGS["facts"])

    user = question

    if search_ctx:
        user += (
            "\n\nAllpool on värske info internetist. "
            "Kasuta seda vastamiseks.\n\n"
            f"{search_ctx}\n\n"
            "Vasta küsimusele selle info põhjal."
        )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


@torch.no_grad()
def generate(tokenizer, model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    out = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("Laen süsteemi...")

    router = load_router()
    tokenizer, model = load_model()
    model = load_adapters(model)
    state = State.load()

    print("Valmis. (exit quit)")

    while True:
        q = input("\nSina: ").strip()
        if q in ["exit", "quit"]:
            break

        topic, conf = router.predict(q)

        # SMART SEARCH LOGIC
        if should_search(q, topic, conf):
            print("[SEARCH ENABLED]")
            search_ctx = search(q)
            topic = "facts"   # override wrong topic
        else:
            search_ctx = None

        if topic not in ADAPTERS:
            topic = "facts"

        print(f"[topic={topic} conf={conf:.2f}]")

        if hasattr(model, "set_adapter"):
            model.set_adapter(topic)

        prompt = build_prompt(tokenizer, q, topic, search_ctx)
        answer = generate(tokenizer, model, prompt)

        print("\nBot:", answer)

        state.last_topic = topic
        state.turn_index += 1
        state.save()


if __name__ == "__main__":
    main()
