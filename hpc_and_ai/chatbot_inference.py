import json
import os
import difflib
import html
import re
import time
import random
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
SEARCH_TIMEOUT = 10
SEARCH_LIMIT = 5
SEARCH_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
SEARCH_CONTACT = os.getenv("SEARCH_CONTACT", "kontakt@example.com")
DISABLE_WIKI_SEARCH = os.getenv("DISABLE_WIKI_SEARCH", "0").lower() in ["1", "true", "yes"]

_last_wikipedia_request_time = 0.0
_wikipedia_request_delay = 1.0
MAX_SEARCH_RPS = int(os.getenv("MAX_SEARCH_RPS", "1"))
_min_wikipedia_interval = 1.0 / MAX_SEARCH_RPS

# Simple on-disk cache to avoid repeated identical queries
SEARCH_CACHE_PATH = Path("search_cache.json")
SEARCH_CACHE_TTL = int(os.getenv("SEARCH_CACHE_TTL", "3600"))
_search_cache: dict = {}

def _load_search_cache() -> None:
    global _search_cache
    try:
        if SEARCH_CACHE_PATH.exists():
            with open(SEARCH_CACHE_PATH, "r", encoding="utf-8") as f:
                _search_cache = json.load(f)
    except Exception:
        _search_cache = {}

def _save_search_cache() -> None:
    try:
        with open(SEARCH_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(_search_cache, f, ensure_ascii=False)
    except Exception:
        pass

# load cache at import
_load_search_cache()
_GLOBAL_WIKI_BLOCK_UNTIL = 0.0

def _save_block_html(query: str, body: str, prefix: str = "search_block") -> None:
    try:
        ts = int(time.time())
        fname = Path(f"{prefix}_{ts}.html")
        with open(fname, "w", encoding="utf-8") as f:
            f.write(f"<!-- query: {query} -->\n")
            f.write(body)
        print(f"[SEARCH] saved blocking HTML to {fname}")
    except Exception:
        pass


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
        "veebist", "search", "kust", "kus", "googli",
        "kes", "mis", "mitu", "kus", "millal", "miks",
    ]

    if any(s in q for s in strong):
        return True

    if confidence < 0.55:
        return True

    return False


def build_search_query(question: str) -> str:
    query = re.sub(r"^(tere|tsau|hei|palun|aitäh)[,\s]+", "", question.strip(), flags=re.IGNORECASE)
    query = re.sub(r"\s+", " ", query)
    return query.strip(" ?!.,")


def _candidate_searxng_urls() -> list[str]:
    candidates = [
        SEARXNG_URL,
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://host.docker.internal:8080",
        "http://searxng:8080",
    ]

    unique: list[str] = []
    seen: set[str] = set()
    for url in candidates:
        normalized = url.rstrip("/")
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(normalized)
    return unique


def _search_searxng(base_url: str, query: str) -> list[str]:
    r = requests.get(
        f"{base_url}/search",
        params={
            "q": query,
            "format": "json",
            "language": "et",
            "safesearch": 2,
        },
        headers={"User-Agent": SEARCH_USER_AGENT},
        timeout=SEARCH_TIMEOUT,
    )
    r.raise_for_status()
    data = r.json()

    results = []
    for item in data.get("results", [])[:SEARCH_LIMIT]:
        title = item.get("title", "").strip()
        content = item.get("content", "").strip() or item.get("snippet", "").strip()
        url = item.get("url", "").strip()
        block = "\n".join(part for part in [title, content, url] if part)
        if block:
            results.append(block)

    return results


def _strip_html(value: str) -> str:
    return re.sub(r"<[^>]+>", "", html.unescape(value)).strip()


def _search_wikipedia(query: str) -> list[str]:
    """Search Wikipedia (Estonian) for topic information with rate limiting."""
    global _last_wikipedia_request_time, _wikipedia_request_delay
    
    if DISABLE_WIKI_SEARCH:
        print("[SEARCH] Wikipedia search disabled via DISABLE_WIKI_SEARCH env var")
        return []
    
    if not query or len(query) < 2:
        return []

    max_retries = 3

    # short-circuit when globally blocked
    global _GLOBAL_WIKI_BLOCK_UNTIL
    if time.time() < _GLOBAL_WIKI_BLOCK_UNTIL:
        print("[SEARCH] Wikipedia temporarily blocked locally; skipping wiki lookup")
        return []

    # check cache first
    try:
        entry = _search_cache.get(query)
        if entry and (time.time() - float(entry.get("ts", 0))) < SEARCH_CACHE_TTL:
            return entry.get("results", [])
    except Exception:
        pass
    for attempt in range(max_retries):
        # enforce both adaptive wikipedia delay and a hard cap from RPS
        effective_interval = max(_wikipedia_request_delay, _min_wikipedia_interval)
        time_since_last_request = time.time() - _last_wikipedia_request_time
        if time_since_last_request < effective_interval:
            time.sleep(effective_interval - time_since_last_request)

        try:
            _last_wikipedia_request_time = time.time()
            
            r = requests.get(
                "https://et.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "srnamespace": 0,
                    "srlimit": 5,
                    "format": "json",
                },
                headers={
                    "User-Agent": SEARCH_USER_AGENT,
                    "From": SEARCH_CONTACT,
                },
                timeout=SEARCH_TIMEOUT,
            )
            r.raise_for_status()
            # If the API responded with HTML (block page) instead of JSON, treat as block
            ctype = r.headers.get("content-type", "").lower()
            if "application/json" not in ctype:
                try:
                    body = r.text
                    _save_block_html(query, body)
                except Exception:
                    pass
                # set a conservative local block interval (5-10 minutes with jitter)
                dur = 300 + random.uniform(0, 120)
                _GLOBAL_WIKI_BLOCK_UNTIL = time.time() + dur
                print(f"[SEARCH] Detected non-JSON response from Wikipedia; pausing wiki lookups for {int(dur)}s")
                return []

            data = r.json()

            results = []
            for item in data.get("query", {}).get("search", []):
                title = item.get("title", "").strip()
                snippet = _strip_html(item.get("snippet", "")).strip()
                if title and snippet and len(snippet) > 10:
                    results.append(f"{title}\n{snippet}")

            if results:
                print(f"[SEARCH] Wikipedia found {len(results)} result(s)")
                try:
                    _search_cache[query] = {"ts": time.time(), "results": results}
                    _save_search_cache()
                except Exception:
                    pass
            else:
                print("[SEARCH] Wikipedia found no results")
            
            return results
            
        except requests.exceptions.HTTPError as exc:
            if exc.response.status_code in (429, 403):
                # Respect server-supplied Retry-After header when present.
                retry_after = None
                try:
                    # log response headers for debugging
                    try:
                        print("[SEARCH] Wikipedia response headers:")
                        for k, v in exc.response.headers.items():
                            print(f"  {k}: {v}")
                    except Exception:
                        pass

                    hdr = exc.response.headers.get("Retry-After")
                    if hdr:
                        # integer seconds
                        if hdr.isdigit():
                            retry_after = float(int(hdr))
                        else:
                            # try HTTP-date parsing
                            from email.utils import parsedate_to_datetime
                            dt = parsedate_to_datetime(hdr)
                            if dt is not None:
                                retry_after = max(0.0, (dt - __import__("datetime").datetime.utcnow()).total_seconds())
                except Exception:
                    retry_after = None
                # compute new adaptive delay: prefer server Retry-After, but never below min interval
                if retry_after is not None:
                    _wikipedia_request_delay = max(retry_after, _min_wikipedia_interval)
                    print(f"[SEARCH] Wikipedia 429/403 received, honoring Retry-After={retry_after}s")
                else:
                    _wikipedia_request_delay = min(max(_wikipedia_request_delay * 1.5, _min_wikipedia_interval), 5.0)
                    print(f"[SEARCH] Wikipedia blocked (rate limit/policy), retry {attempt+1}/{max_retries}, backoff={_wikipedia_request_delay}s")

                # Save blocking HTML body if available
                try:
                    body = exc.response.text
                    _save_block_html(query, body)
                except Exception:
                    pass

                # set a conservative local block interval (5 minutes + jitter)
                block_dur = max(60, int(_wikipedia_request_delay * 60))
                block_dur = 300 + random.uniform(0, 120)
                _GLOBAL_WIKI_BLOCK_UNTIL = time.time() + block_dur
                print(f"[SEARCH] Local block set for {int(block_dur)}s due to repeated 4xx responses")

                if attempt < max_retries - 1:
                    time.sleep(min(_wikipedia_request_delay, 2 ** attempt))
                    continue
            print(f"[SEARCH] Wikipedia error: {exc}")
            return []
        except Exception as exc:
            print(f"[SEARCH] Wikipedia error: {exc}")
            return []
    
    print("[SEARCH] Wikipedia max retries exceeded")
    return []


def _search_weather(query: str) -> list[str]:
    """Try to extract weather-related info from query."""
    if not any(w in query.lower() for w in ["ilm", "temp", "weather", "rain", "snow"]):
        return []

    cities = {
        "tallinn": (59.4370, 24.7536),
        "tartu": (58.3806, 26.7219),
        "pärnu": (58.3853, 24.5014),
        "haapsalu": (58.9455, 23.5447),
        "narva": (59.3742, 28.1948),
        "kuressaare": (58.2548, 22.4898),
        "rakvere": (59.3528, 26.3597),
        "kärdla": (59.0097, 23.1940),
        "viljandi": (58.3645, 25.5889),
        "põltsamaa": (58.7514, 25.9706),
        "estonia": (58.5953, 25.0136),
        "tallinna": (59.4370, 24.7536),
        "tallinnas": (59.4370, 24.7536),
    }

    lat, lon = None, None
    query_lower = query.lower()

    for city_name, coords in cities.items():
        if city_name in query_lower:
            lat, lon = coords
            break

    if lat is None or lon is None:
        return []

    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,weather_code,wind_speed_10m",
                "temperature_unit": "celsius",
                "timezone": "Europe/Tallinn",
            },
            timeout=SEARCH_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()

        current = data.get("current", {})
        temp = current.get("temperature_2m", "?")
        wind = current.get("wind_speed_10m", "?")
        weather_code = current.get("weather_code", 0)

        conditions = {
            0: "selge",
            1: "peaaegu selge",
            2: "osaliselt pilvine",
            3: "pilvine",
            45: "udu",
            48: "külmuva udu",
            51: "kerge vihmane",
            61: "vihmane",
            71: "kerge lumesadu",
            81: "intensiivne vihm",
        }

        condition = conditions.get(weather_code, "tundmatu")

        city_found = None
        for city_name, coords in cities.items():
            if coords == (lat, lon):
                city_found = city_name.replace("a", "").replace("s", "").title()
                break

        if not city_found:
            city_found = "Otsitud kohas"

        result = f"{city_found} ilm: {temp}°C, {condition}, tuul {wind} m/s"
        return [result]

    except Exception as exc:
        print(f"[SEARCH] Weather API error: {exc}")
        return []


def search(query: str) -> str:
    cleaned_query = build_search_query(query)
    last_searxng_error = None

    # Skip external search if Wikipedia search is disabled (IP likely rate-limited)
    if not DISABLE_WIKI_SEARCH:
        for base_url in _candidate_searxng_urls():
            try:
                results = _search_searxng(base_url, cleaned_query)
                if results:
                    print(f"[SEARCH] searxng -> {base_url}")
                    return "\n\n".join(results)
            except Exception as exc:
                last_searxng_error = exc

        if last_searxng_error is not None:
            print(f"[SEARCH] searxng unavailable, using fallback: {last_searxng_error}")

    weather_results = _search_weather(cleaned_query)
    if weather_results:
        print("[SEARCH] fallback -> weather API")
        return "\n\n".join(weather_results)

    wiki_results = _search_wikipedia(cleaned_query)
    if wiki_results:
        print("[SEARCH] fallback -> wikipedia API")
        return "\n\n".join(wiki_results)

    if DISABLE_WIKI_SEARCH:
        print("[SEARCH] external search disabled (DISABLE_WIKI_SEARCH=1), using internal knowledge only")
    else:
        print("[SEARCH] all methods failed, using only internal knowledge")
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