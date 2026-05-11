from __future__ import annotations

import json
import os
import difflib
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any
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

#------------------------------------------------------------------------------
# RUUTERI PATHID
#------------------------------------------------------------------------------
# DOCKERI PATHID
ROUTER_PATHS = [
    "adapters/2_290426/router_artifacts/router_context.joblib",
    "adapters/2_290426/router_artifacts/router_baseline.joblib",
]
# HPC PATHID
"""
SIIA TULEVAD HPC PATHID
"""

# LOCAL PATHID
"""
ROUTER_PATHS = [
    "../adapters/2_290426/router_artifacts/router_context.joblib",
    "../adapters/2_290426/router_artifacts/router_baseline.joblib",
]
"""

#------------------------------------------------------------------------------
# ADAPTERI PATHID
#------------------------------------------------------------------------------
"""
ADAPTERS = {
    "facts": "adapters/2_290426/facts",
    "kasitoo": "adapters/2_290426/kasitoo",
    "luuletused": "adapters/2_290426/luuletused",
    "muistendid": "adapters/2_290426/muistendid",
    "tahtpaevad": "adapters/2_290426/tahtpaevad",
}
"""

# HPC JA LOCAL PATHID
ADAPTERS = {
    "facts": "adapters/2_290426/facts",
    "kasitoo": "adapters/2_290426/kasitoo",
    "luuletused": "adapters/2_290426/luuletused",
    "muistendid": "adapters/2_290426/muistendid",
    "tahtpaevad": "adapters/2_290426/tahtpaevad",
}


SYSTEM_MSGS = {
    "facts": "Sa oled abivalmis assistent. Vasta lühidalt ja eesti keeles. Teema on huvitavad faktid",
    "kasitoo": "Sa oled abivalmis assistent. Vasta lühidalt ja eesti keeles. Teema on käsitöö.",
    "luuletused": "Sa oled abivalmis assistent. Vasta lühidalt ja eesti keeles. Teema on luuletused.",
    "tahtpaevad": "Sa oled abivalmis assistent. Vasta lühidalt ja eesti keeles. Teema on tähtpäevad.",
    "muistendid": "Sa oled abivalmis assistent. Vasta lühidalt ja eesti keeles. Teema on muistendid.",
}

STATE_PATH = Path("conversation_state.json")
MAX_RECENT_MESSAGES = 12
MAX_RECENT_FOR_PROMPT = 6


BASE_SYSTEM = (
    "Sa oled eesti keeles töötav assistent.\n"
    "Kasuta ainult antud konteksti.\n"
    "Kui kasutaja küsib jätkamist, jätka viimast asjakohast vastust.\n"
    "Ära leiuta uut infot.\n"
    "Vasta lühidalt ja selgelt."
)

SEARXNG_URL = os.getenv("SEARXNG_URL", "http://searxng:8080")


# =============================================================================
# STATE
# =============================================================================

@dataclass
class State:
    last_topic: str = "unknown"
    last_answer: str = ""
    last_user: str = ""
    conversation_summary: str = ""
    recent_messages: list[dict[str, str]] = field(default_factory=list)
    turn_index: int = 1

    def save(self)  -> None:
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls) -> "State":
        if STATE_PATH.exists():
            try:
                with open(STATE_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return cls(
                    last_topic=data.get("last_topic", "unknown"),
                    last_answer=data.get("last_answer", ""),
                    last_user=data.get("last_user", ""),
                    conversation_summary=data.get("conversation_summary", ""),
                    recent_messages=data.get("recent_messages", []),
                    turn_index=int(data.get("turn_index", 1)),
                )
            except Exception:
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

    def predict(self, text: str) -> tuple[str, float]:
        emb = self.embedder.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        probs = self.model.predict_proba(emb)[0]
        idx = int(np.argmax(probs))
        return self.model.classes_[idx], float(probs[idx])


def load_router():
    for p in ROUTER_PATHS:
        if Path(p).exists():
            return Router(p)
    raise FileNotFoundError("Router puudub")


# =============================================================================
# CONTINUATION DETECTION
# =============================================================================

def is_continuation(text: str) -> bool:
    q = text.lower().strip()

    if not q:
        return False

    # Exact-ish short continuation commands
    patterns = [
        r"^jätka$",
        r"^jätka palun$",
        r"^palun jätka$",
        r"^edasi$",
        r"^räägi veel$",
        r"^veel$",
        r"^anna edasi$",
        r"^continue$",
        r"^go on$",
        r"^more$",
    ]
    for pattern in patterns:
        if re.match(pattern, q):
            return True

    # Loose phrase containment
    phrases = ["jätka", "edasi", "räägi veel", "anna edasi", "continue", "go on"]
    for p in phrases:
        if p in q:
            return True
        if difflib.SequenceMatcher(None, q, p).ratio() > 0.80:
            return True

    return False


# =============================================================================
# MODEL + ADAPTERS
# =============================================================================
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    model_cls = Gemma3ForCausalLM if Gemma3ForCausalLM else AutoModelForCausalLM
    model = model_cls.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)

    model.eval()
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
# SEARCH
# =============================================================================

def should_search(question: str, topic: str, confidence: float, continuation: bool) -> bool:
    if continuation:
        return False

    q = question.lower().strip()

    strong = [
        "ilm", "täna", "praegu", "mis kell",
        "uudised", "internet", "otsi", "otsi internetist",
        "veebist", "search", "kust", "kus", "googli"
    ]

    if any(s in q for s in strong):
        return True

    if confidence < 0.55:
        return True

    return False


def search(query):
    try:
        r = requests.get(
            f"{SEARXNG_URL}/search",
            params={
                "q": query,
                "format": "json",
                "language": "et",
                "safesearch": 2,
            },
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()

        results = []
        for item in data.get("results", [])[:5]:
            results.append(f"{item.get('title','')}\n{item.get('content','')}")

        return "\n\n".join(results)

    except Exception as e:
        print(f"[SEARCH ERROR] {e}")
        return ""


# =============================================================================
# PROMPT BUILDING
# =============================================================================

def format_summary(summary: Any) -> str:
    if not summary:
        return "puudub"
    if isinstance(summary, dict):
        lines = []
        for k, v in summary.items():
            if v:
                lines.append(f"- {k}: {v}")
        return "\n".join(lines) if lines else "puudub"
    return str(summary)


def format_recent_messages(recent_messages: list[dict[str, str]] | None) -> str:
    if not recent_messages:
        return "puuduvad"

    lines = []
    for msg in recent_messages[-MAX_RECENT_FOR_PROMPT:]:
        role = msg.get("role", "").upper().strip()
        content = msg.get("content", "").strip()
        if role and content:
            lines.append(f"{role}: {content}")

    return "\n".join(lines) if lines else "puuduvad"


# =============================================================================
# BUILD PROMPT
# =============================================================================

def build_prompt(tokenizer, question: str, topic: str, *, summary: Any = None,
    recent_messages: list[dict[str, str]] | None = None, search_ctx: str | None = None,
    continuation: bool = False, last_answer: str = "", last_user: str = "",):

    topic_system = SYSTEM_MSGS.get(topic, SYSTEM_MSGS["facts"])
    system = f"{BASE_SYSTEM}\n\nTeemaspetsiifiline lisareegel: {topic_system}"

    summary_text = format_summary(summary)
    recent_text = format_recent_messages(recent_messages)


    #NB PRAEGU POLE TREENIMATRJALIDES SELLIST TASK-I
    if continuation:
        task = (
            "Jätka eelmist vastust.\n"
            "Kasuta viimast kasutaja sõnumit, viimast vastust ja hiljutisi sõnumeid.\n"
            "Ära muuda teemat ega leiuta uut infot."
        )

    else:
        task = question.strip()

    user_parts = [
        "[TEEMA]",
        topic,
        "",
        "[KOKKUVÕTE]",
        summary_text,
        "",
        "[VIIMASED SÕNUMID]",
        recent_text,
        "",
        "[EELMINE KASUTAJA SÕNUM]",
        last_user.strip() or "puudub",
        "",
        "[EELMINE VASTUS]",
        last_answer.strip() or "puudub",
    ]

    if search_ctx:
        user_parts += [
            "",
            "[OTSING]",
            search_ctx.strip(),
        ]

    user_parts += [
        "",
        "[ÜLESANNE]",
        task,
    ]

    user = "\n".join(user_parts)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# =============================================================================
# GENERATION
# =============================================================================
@torch.no_grad()
def generate(tokenizer, model, prompt: str) -> str:
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    out = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=True,
        temperature=0.30,
        top_p=0.80,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

# =============================================================================
# HELPERS
# =============================================================================

def update_recent_messages(state: State, user_text: str, assistant_text: str) -> None:
    state.recent_messages.extend(
        [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]
    )
    state.recent_messages = state.recent_messages[-MAX_RECENT_MESSAGES:]


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
        if q.lower() in ["exit", "quit"]:
            break

        continuation = is_continuation(q)
        if continuation and state.last_topic != "unknown":
            topic = state.last_topic
            conf = 1.0
        else:
            topic, conf = router.predict(q)

        # SMART SEARCH LOGIC
        if should_search(q, topic, conf, continuation):
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

        prompt = build_prompt(
            tokenizer,
            q,
            topic,
            summary=state.conversation_summary,
            recent_messages=state.recent_messages,
            search_ctx=search_ctx,
            continuation=continuation,
            last_answer=state.last_answer,
            last_user=state.last_user,
        )

        answer = generate(tokenizer, model, prompt)

        print("\nBot:", answer)

        state.last_topic = topic
        state.last_user = q
        state.last_answer = answer
        update_recent_messages(state, q, answer)
        state.turn_index += 1
        state.save()


if __name__ == "__main__":
    main()