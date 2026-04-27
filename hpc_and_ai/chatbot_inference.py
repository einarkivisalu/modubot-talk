from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import requests
import torch
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from content_policy import is_blocked_topic, refusal_text

try:
    from transformers import Gemma3ForCausalLM  # type: ignore
except Exception:  # pragma: no cover
    Gemma3ForCausalLM = None  # type: ignore


# =============================================================================
# Config
# =============================================================================

MODEL_ID = os.getenv("MODEL_ID", "google/gemma-3-1b-it")

ROUTER_MODEL_CANDIDATES = [
    os.getenv("ROUTER_MODEL_PATH", "router_artifacts/router_context.joblib"),
    "router_artifacts/router_baseline.joblib",
]

SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8080")
STATE_PATH = Path(os.getenv("STATE_PATH", "conversation_state.json"))

ROUTER_THRESHOLD = float(os.getenv("ROUTER_THRESHOLD", "0.45"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "220"))

ADAPTER_CANDIDATES: Dict[str, List[str]] = {
    "facts": [
        "adapters/facts",
        "adapters/huvitavad_faktid",
        "adapters/faktid",
    ],
    "kasitoo": [
        "adapters/kasitoo",
        "adapters/käsitöö",
    ],
    "kunstiajalugu": [
        "adapters/kunstiajalugu",
        "adapters/kunsti_ajalugu",
    ],
    "luuletused": [
        "adapters/luuletused",
    ],
    "muistendid": [
        "adapters/muistendid",
        "adapters/muistendit",
    ],
    "tahtpaevad": [
        "adapters/tahtpaevad",
        "adapters/tahtpäevad",
    ],
}

DEFAULT_TOPIC = "facts"

SEARCH_HINTS = [
    "praegu",
    "täna",
    "homme",
    "eelmine",
    "viimane",
    "uus",
    "uudis",
    "uudised",
    "internet",
    "otsi",
    "kust",
    "mis kell",
    "ilm",
    "hetkel",
]

FOLLOW_UP_HINTS = [
    "veel",
    "selgita",
    "jätka",
    "täpsemalt",
    "see",
    "selle",
    "sellest",
    "seda",
    "tema",
    "nendest",
]

SUMMARY_REFRESH_EVERY = 3
SUMMARY_RECENT_TURNS = 8


# =============================================================================
# Runtime state
# =============================================================================

@dataclass
class ConversationState:
    last_topic: str = "unknown"
    turn_index: int = 1
    is_follow_up: int = 0
    user_name: str = ""
    last_search_used: bool = False

    last_user_text: str = ""
    last_assistant_text: str = ""
    summary: str = ""
    recent_turns: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationState":
        return cls(
            last_topic=data.get("last_topic", "unknown"),
            turn_index=int(data.get("turn_index", 1)),
            is_follow_up=int(data.get("is_follow_up", 0)),
            user_name=data.get("user_name", ""),
            last_search_used=bool(data.get("last_search_used", False)),
            last_user_text=data.get("last_user_text", ""),
            last_assistant_text=data.get("last_assistant_text", ""),
            summary=data.get("summary", ""),
            recent_turns=list(data.get("recent_turns", [])) if data.get("recent_turns") else [],
        )


def load_state(path: Path = STATE_PATH) -> ConversationState:
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                return ConversationState.from_dict(json.load(f))
        except Exception:
            pass
    return ConversationState()


def save_state(state: ConversationState, path: Path = STATE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(state.to_dict(), f, ensure_ascii=False, indent=2)


def append_turn(state: ConversationState, role: str, text: str) -> None:
    state.recent_turns.append({"role": role, "text": text})
    if len(state.recent_turns) > SUMMARY_RECENT_TURNS:
        state.recent_turns = state.recent_turns[-SUMMARY_RECENT_TURNS:]


def detect_follow_up(text: str) -> int:
    t = text.lower()
    return int(any(h in t for h in FOLLOW_UP_HINTS))


# =============================================================================
# SearXNG client
# =============================================================================

class SearxngClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def search(
        self,
        query: str,
        language: str = "et",
        safesearch: int = 2,
        limit: int = 5,
    ) -> List[dict]:
        params = {
            "q": query,
            "format": "json",
            "language": language,
            "safesearch": safesearch,
        }
        url = f"{self.base_url}/search"
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()

        results = data.get("results", [])[:limit]
        cleaned = []
        for item in results:
            cleaned.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("content", "") or item.get("snippet", "") or "",
                }
            )
        return cleaned


def build_search_context(results: List[dict]) -> str:
    if not results:
        return "Otsing ei andnud usaldusväärseid tulemusi."

    blocks = []
    for i, r in enumerate(results, start=1):
        blocks.append(
            f"[{i}] {r['title']}\n"
            f"{r['snippet']}\n"
            f"{r['url']}"
        )
    return "\n\n".join(blocks)


# =============================================================================
# Router loader
# =============================================================================

def resolve_existing_path(candidates: List[str]) -> Optional[Path]:
    for c in candidates:
        p = Path(c)
        if p.exists():
            return p
    return None


class RuntimeTopicRouter:
    def __init__(self, model_path: str):
        payload = joblib.load(model_path)
        self.embedding_model_name = payload["embedding_model_name"]
        self.use_context = bool(payload.get("use_context", False))
        self.classifier = payload["classifier"]
        self.topic_encoder = payload.get("topic_encoder", None)
        self.num_scaler = payload.get("num_scaler", None)
        self.embedder = SentenceTransformer(self.embedding_model_name)

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        return self.embedder.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def _numeric_matrix(self, rows: List[dict]) -> np.ndarray:
        return np.array(
            [
                [
                    float(r.get("turn_index", 1)),
                    float(r.get("is_follow_up", 0)),
                ]
                for r in rows
            ],
            dtype=np.float32,
        )

    def _transform(self, rows: List[dict]) -> np.ndarray:
        texts = [r["text"] for r in rows]
        x_text = self._embed_texts(texts)

        if not self.use_context:
            return x_text

        if self.topic_encoder is None or self.num_scaler is None:
            raise RuntimeError("Context router expects topic_encoder and num_scaler in the joblib payload.")

        last_topics = np.array([[r.get("last_topic", "unknown")] for r in rows], dtype=object)
        x_topic = self.topic_encoder.transform(last_topics)
        x_num = self.num_scaler.transform(self._numeric_matrix(rows))
        return np.hstack([x_text, x_topic, x_num])

    def predict_one(
        self,
        text: str,
        last_topic: str = "unknown",
        turn_index: int = 1,
        is_follow_up: int = 0,
        threshold: float = ROUTER_THRESHOLD,
    ) -> Tuple[str, float]:
        row = {
            "text": text,
            "last_topic": last_topic,
            "turn_index": turn_index,
            "is_follow_up": is_follow_up,
        }
        x = self._transform([row])

        proba = self.classifier.predict_proba(x)[0]
        classes = self.classifier.classes_
        best_idx = int(np.argmax(proba))
        best_topic = str(classes[best_idx])
        confidence = float(proba[best_idx])

        if confidence < threshold:
            return "fallback", confidence
        return best_topic, confidence


def load_router() -> RuntimeTopicRouter:
    path = resolve_existing_path(ROUTER_MODEL_CANDIDATES)
    if path is None:
        raise FileNotFoundError(
            "Ei leidnud routeri .joblib faili. Treeni router enne valmis "
            "(nt router_artifacts/router_context.joblib või router_artifacts/router_baseline.joblib)."
        )
    return RuntimeTopicRouter(str(path))


# =============================================================================
# Model + adapters
# =============================================================================

def load_base_model_and_tokenizer(model_id: str):
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        token=token if token else None,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

    if Gemma3ForCausalLM is not None:
        try:
            model = Gemma3ForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                trust_remote_code=True,
                ignore_mismatched_sizes=True,
                low_cpu_mem_usage=True,
                token=token if token else None,
            )
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                trust_remote_code=True,
                ignore_mismatched_sizes=True,
                low_cpu_mem_usage=True,
                token=token if token else None,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            trust_remote_code=True,
            ignore_mismatched_sizes=True,
            low_cpu_mem_usage=True,
            token=token if token else None,
        )

    model.config.use_cache = False
    model.resize_token_embeddings(len(tokenizer))
    return tokenizer, model


def load_model_with_adapters(model_id: str, topic_to_candidates: Dict[str, List[str]]):
    tokenizer, base_model = load_base_model_and_tokenizer(model_id)

    loaded_topics: Dict[str, str] = {}
    first_topic: Optional[str] = None
    first_path: Optional[Path] = None

    for topic, candidates in topic_to_candidates.items():
        path = resolve_existing_path(candidates)
        if path is not None:
            loaded_topics[topic] = str(path)
            if first_topic is None:
                first_topic = topic
                first_path = path

    model = base_model
    if first_topic is not None and first_path is not None:
        model = PeftModel.from_pretrained(
            base_model,
            str(first_path),
            adapter_name=first_topic,
        )
        for topic, path_str in loaded_topics.items():
            if topic == first_topic:
                continue
            model.load_adapter(path_str, adapter_name=topic)

    return tokenizer, model, loaded_topics


# =============================================================================
# Prompt / summary / search heuristics
# =============================================================================

def should_search(question: str, topic: str, confidence: float) -> bool:
    q = question.lower()
    if topic == "fallback" or confidence < ROUTER_THRESHOLD:
        return True
    return any(h in q for h in SEARCH_HINTS)


def build_prompt(
    tokenizer,
    question: str,
    topic: str,
    state: ConversationState,
    search_context: Optional[str] = None,
) -> str:
system_msg = (
    "Sa oled kokkuvõtete kirjutaja. "
    "Kirjuta väga lühike kokkuvõte (1-2 lauset). "
    "Ära lisa uut infot. Ära korda sõnu. "
    "Kasuta lihtsat eesti keelt."
)

    user_block = [
        f"Kasutaja küsimus: {question}",
        f"Valitud teema: {topic}",
        f"Eelmine teema: {state.last_topic}",
        f"Vestluse pööre: {state.turn_index}",
    ]

    if state.summary:
        user_block.append("")
        user_block.append("Vestluse kokkuvõte:")
        user_block.append(state.summary)

    if state.recent_turns:
        user_block.append("")
        user_block.append("Viimased pöörded:")
        for turn in state.recent_turns[-SUMMARY_RECENT_TURNS:]:
            user_block.append(f"{turn['role']}: {turn['text']}")

    if search_context:
        user_block.append("")
        user_block.append("Otsingukontekst:")
        user_block.append(search_context)

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": "\n".join(user_block)},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass

    return (
        f"SYSTEM: {system_msg}\n\n"
        f"USER:\n" + "\n".join(user_block) + "\n\n"
        f"ASSISTANT:"
    )


def build_summary_prompt(state: ConversationState) -> str:
    system_msg = (
        "Sa oled kokkuvõtete kirjutaja. "
        "Sinu ülesanne on uuendada vestluse lühikokkuvõtet. "
        "Kirjuta ainult 1-3 lühikest lauset eesti keeles. "
        "Maini teemasid, kasutaja eelistusi ja olulisi fakte, kui need on olemas."
    )

    recent_text = []
    for turn in state.recent_turns[-SUMMARY_RECENT_TURNS:]:
        recent_text.append(f"{turn['role']}: {turn['text']}")

    user_msg = (
        f"Senine kokkuvõte:\n{state.summary or '(puudub)'}\n\n"
        f"Viimased pöörded:\n" + "\n".join(recent_text) + "\n\n"
        "Kirjuta uus, lühike ja kompaktne kokkuvõte."
    )

    return f"SYSTEM: {system_msg}\n\nUSER: {user_msg}\n\nASSISTANT:"


@torch.inference_mode()
def generate_text(tokenizer, model, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.2,
        repetition_penalty=1.08,
        pad_token_id=tokenizer.eos_token_id,
    )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return text


def refresh_summary_if_needed(tokenizer, model, state: ConversationState) -> str:
    if not state.summary and len(state.recent_turns) < 4:
        return state.summary

    if state.turn_index % SUMMARY_REFRESH_EVERY != 0 and state.summary:
        return state.summary

    prompt = build_summary_prompt(state)
    summary = generate_text(tokenizer, model, prompt, max_new_tokens=90).strip()

    summary = summary.replace("Kokkuvõte:", "").strip()
    return summary


def update_state_after_turn(
    state: ConversationState,
    user_text: str,
    assistant_text: str,
    topic: str,
    search_used: bool,
    tokenizer,
    model,
) -> ConversationState:
    state.last_topic = topic
    state.turn_index += 1
    state.is_follow_up = detect_follow_up(user_text)
    state.last_user_text = user_text
    state.last_assistant_text = assistant_text
    state.last_search_used = search_used

    append_turn(state, "user", user_text)
    append_turn(state, "assistant", assistant_text)

    state.summary = refresh_summary_if_needed(tokenizer, model, state)
    return state


# =============================================================================
# Main orchestration
# =============================================================================

def answer_once(
    user_text: str,
    state: ConversationState,
    router: RuntimeTopicRouter,
    tokenizer,
    model,
    searx: SearxngClient,
    loaded_topics: Dict[str, str],
) -> Tuple[str, ConversationState, dict]:
    # 1) Enne kõike: sisufilter.
    blocked, blocked_type = is_blocked_topic(user_text)
    if blocked:
        answer = refusal_text(blocked_type)

        state = update_state_after_turn(
            state=state,
            user_text=user_text,
            assistant_text=answer,
            topic=blocked_type or "blocked",
            search_used=False,
            tokenizer=tokenizer,
            model=model,
        )
        save_state(state)

        return answer, state, {
            "topic": blocked_type or "blocked",
            "confidence": 1.0,
            "search_used": False,
            "search_results": [],
        }

    # 2) Kui pole keelatud teema, siis tavapärane router.
    topic, confidence = router.predict_one(
        text=user_text,
        last_topic=state.last_topic,
        turn_index=state.turn_index,
        is_follow_up=state.is_follow_up,
    )

    if topic == "fallback":
        topic = DEFAULT_TOPIC if DEFAULT_TOPIC in loaded_topics or not loaded_topics else DEFAULT_TOPIC

    adapter_topic = topic if topic in loaded_topics else None

    search_context = None
    search_results = []
    search_used = False

    # 3) Kui vaja, otsi netist.
    if should_search(user_text, topic, confidence):
        try:
            search_results = searx.search(
                query=user_text,
                language="et",
                safesearch=2,
                limit=5,
            )
            search_context = build_search_context(search_results)
            search_used = True
        except Exception as e:
            search_context = f"Otsing ebaõnnestus: {e}"
            search_used = False

    # 4) Aktiveeri õige adapter.
    if hasattr(model, "set_adapter") and adapter_topic is not None:
        try:
            model.set_adapter(adapter_topic)
        except Exception:
            pass

    # 5) Genereeri vastus.
    prompt = build_prompt(
        tokenizer=tokenizer,
        question=user_text,
        topic=topic,
        state=state,
        search_context=search_context,
    )

    answer = generate_text(tokenizer, model, prompt, max_new_tokens=MAX_NEW_TOKENS)

    # 6) Uuenda state.
    state = update_state_after_turn(
        state=state,
        user_text=user_text,
        assistant_text=answer,
        topic=topic,
        search_used=search_used,
        tokenizer=tokenizer,
        model=model,
    )

    meta = {
        "topic": topic,
        "confidence": confidence,
        "search_used": search_used,
        "search_results": search_results,
    }
    return answer, state, meta


def main():
    print("Laen ruuteri...")
    router = load_router()

    print("Laen mudeli ja adapterid...")
    tokenizer, model, loaded_topics = load_model_with_adapters(MODEL_ID, ADAPTER_CANDIDATES)

    if not loaded_topics:
        print("Hoiatus: ühtegi adapterit ei leitud. Mudel töötab ilma adapteriteta.")

    print("Laen SearXNG kliendi...")
    searx = SearxngClient(SEARXNG_URL)

    state = load_state()
    print("Valmis. Kirjuta 'exit' või 'quit' lõpetamiseks.\n")

    while True:
        try:
            user_text = input("Sina: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nLõpetan.")
            break

        if not user_text:
            continue

        if user_text.lower() in {"exit", "quit"}:
            break

        answer, state, meta = answer_once(
            user_text=user_text,
            state=state,
            router=router,
            tokenizer=tokenizer,
            model=model,
            searx=searx,
            loaded_topics=loaded_topics,
        )

        save_state(state)

        print(f"\n[topic={meta['topic']} | conf={meta['confidence']:.2f} | search={meta['search_used']}]")
        print(f"Bot: {answer}\n")


if __name__ == "__main__":
    main()
