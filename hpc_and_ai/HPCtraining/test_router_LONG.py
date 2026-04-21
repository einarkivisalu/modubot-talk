# test_router_LONG.py
import torch
from pathlib import Path

from router_VERA import TopicRouter
from transformers import AutoTokenizer

try:
    from transformers import Gemma3ForCausalLM
except Exception:
    Gemma3ForCausalLM = None

from peft import PeftModel


BASE_MODEL_ID = "google/gemma-3-1b-it"
BASE_DIR = Path(__file__).resolve().parent
ROUTER_PATH = "/gpfs/mariana/home/anemoo/router_artifacts/topic_router.joblib"

ADAPTER_DIRS = {
    "facts": Path("/gpfs/mariana/home/anemoo/adapters1.1/facts"),
    "kasitoo": Path("/gpfs/mariana/home/anemoo/adapters1.1/kasitoo"),
    "luuletused": Path("/gpfs/mariana/home/anemoo/adapters1.1/luuletused"),
    "muistendid": Path("/gpfs/mariana/home/anemoo/adapters1.1/muistendid"),
    "tahtpaevad": Path("/gpfs/mariana/home/anemoo/adapters1.1/tahtpaevad"),
}
SYSTEM_MSGS = {
    "facts": "Sa oled abivalmis assistent. Vasta lühidalt ja eesti keeles. Teema on huvitavad faktid.",
    "kasitoo": "Sa oled abivalmis assistent. Vasta lühidalt ja eesti keeles. Teema on käsitöö.",
    "luuletused": "Sa oled abivalmis assistent. Vasta lühidalt ja eesti keeles. Teema on luuletused.",
    "tahtpaevad": "Sa oled abivalmis assistent. Vasta lühidalt ja eesti keeles. Teema on tähtpäevad.",
    "muistendid": "Sa oled abivalmis assistent. Vasta lühidalt ja eesti keeles. Teema on muistendid.",
}

MODEL = None
TOKENIZER = None


def load_base_model():
    global MODEL, TOKENIZER

    if MODEL is not None and TOKENIZER is not None:
        return TOKENIZER, MODEL

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA pole saadaval. Kontrolli PyTorch/CUDA installi.")

    print("Laen tokenizeri...", flush=True)
    TOKENIZER = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    if TOKENIZER.pad_token_id is None:
        TOKENIZER.pad_token = TOKENIZER.eos_token

    print("Laen base tekstimudeli...", flush=True)

    if Gemma3ForCausalLM is not None:
        base_model = Gemma3ForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
    else:
        from transformers import AutoModelForCausalLM

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )

    base_model = base_model.to("cuda")
    base_model.config.use_cache = True

    MODEL = base_model
    return TOKENIZER, MODEL


def preload_adapters():
    tokenizer, model = load_base_model()

    print("Laen adapterid ette...", flush=True)
    first_topic = True

    for topic, adapter_dir in ADAPTER_DIRS.items():
        if not adapter_dir.is_dir():
            raise FileNotFoundError(f"Puudub adapteri kaust: {adapter_dir}")

        print(f"  -> {topic}: {adapter_dir}", flush=True)

        if first_topic:
            model = PeftModel.from_pretrained(
                model,
                str(adapter_dir),
                adapter_name=topic,
                is_trainable=False,
            )
            first_topic = False
        else:
            model.load_adapter(
                str(adapter_dir),
                adapter_name=topic,
                is_trainable=False,
            )

    model.eval()

    global MODEL, TOKENIZER
    MODEL = model
    return TOKENIZER, MODEL


def generate_answer(topic, question):
    tokenizer, model = load_base_model()

    if topic not in ADAPTER_DIRS:
        raise ValueError(f"Tundmatu teema: {topic}")

    model.set_adapter(topic)

    messages = [
        {"role": "system", "content": SYSTEM_MSGS[topic]},
        {"role": "user", "content": question},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    print("Genereerin vastust...", flush=True)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    generated_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def looks_like_follow_up(question: str) -> bool:
    q = question.lower().strip()
    return (
        "räägi veel sellest" in q
        or "räägi sellest" in q
        or "mis edasi" in q
        or "ja siis" in q
        or "selle kohta" in q
    )


def router_smoke_test():
    router = TopicRouter(ROUTER_PATH)

    topic, conf = router.predict(
        "Räägi veel sellest.",
        last_topic="tahtpaevad",
        turn_index=2,
        is_follow_up=1,
    )

    print(f"Smoke test -> topic: {topic}, conf: {conf:.3f}")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA pole saadaval. Kontrolli PyTorch/CUDA installi.")

    # Eelkontroll, et uus ruuter töötab õigesti
    router_smoke_test()

    router = TopicRouter(ROUTER_PATH)
    preload_adapters()

    print("Sisesta küsimus. Väljumiseks kirjuta exit või quit.")

    last_topic = "unknown"
    turn_index = 1

    while True:
        question = input("\nKüsimus> ").strip()
        if not question or question.lower() in {"exit", "quit"}:
            break

        is_follow_up = 1 if looks_like_follow_up(question) else 0

        if is_follow_up and last_topic != "unknown":
            topic = last_topic
            conf = 1.0
            probs = [(topic, 1.0)]
            used_fallback = False
        else:
            topic, conf = router.predict(
                text=question,
                last_topic="unknown",
                turn_index=turn_index,
                is_follow_up=0,
                threshold=0.45,
            )

            probs = router.predict_with_probs(
                text=question,
                last_topic="unknown",
                turn_index=turn_index,
                is_follow_up=0,
            )

            used_fallback = (topic == "fallback")
            if used_fallback:
                print("Confidence oli madal, kasutan fallback teemana: facts")
                topic = "facts"

        print(f"Router: {topic}  confidence={conf:.3f}")
        print("Probs:", probs[:3])

        answer = generate_answer(topic, question)
        print("\nVastus:\n")
        print(answer)

        # follow up loogika
        if not is_follow_up and not used_fallback:
            last_topic = topic

        turn_index += 1
if __name__ == "__main__":
    main()
