import os
import joblib
import torch

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL_ID = "google/gemma-3-1b-it"
ROUTER_PATH = "router_artifacts/topic_router.joblib"

ADAPTER_DIRS = {
    "facts": "adapters/facts",
    "kasitoo": "adapters/kasitoo",
    "luuletused": "adapters/luuletused",
}

SYSTEM_MSGS = {
    "facts": "Sa oled abivalmis assistent. Vasta lühidalt ja eesti keeles. Teema on huvitavad faktid.",
    "kasitoo": "Sa oled abivalmis assistent. Vasta lühidalt ja eesti keeles. Teema on käsitöö.",
    "luuletused": "Sa oled abivalmis assistent. Vasta lühidalt ja eesti keeles. Teema on luuletused.",
}

MODEL = None
TOKENIZER = None


def load_router():
    if not os.path.isfile(ROUTER_PATH):
        raise FileNotFoundError(f"Puudub router fail: {ROUTER_PATH}")
    return joblib.load(ROUTER_PATH)


def route_question(question, embedder, clf, threshold=0.45):
    x = embedder.encode([question], normalize_embeddings=True, show_progress_bar=False)
    proba = clf.predict_proba(x)[0]
    labels = clf.classes_

    best_idx = int(proba.argmax())
    best_topic = labels[best_idx]
    best_conf = float(proba[best_idx])

    probs = {label: float(p) for label, p in zip(labels, proba)}
    below_threshold = best_conf < threshold

    return best_topic, best_conf, probs, below_threshold


def load_base_model():
    global MODEL, TOKENIZER

    if MODEL is not None and TOKENIZER is not None:
        return TOKENIZER, MODEL

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA pole saadaval.")

    print("Laen tokenizeri...", flush=True)
    TOKENIZER = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    if TOKENIZER.pad_token_id is None:
        TOKENIZER.pad_token = TOKENIZER.eos_token

    print("Laen base mudeli...", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    base_model.config.use_cache = True
    MODEL = base_model
    return TOKENIZER, MODEL


def preload_adapters():
    """
    Laeb adapterid samale base mudelile.
    Esimene adapter läheb läbi PeftModel.from_pretrained,
    järgmised lisatakse load_adapter abil.
    """
    tokenizer, model = load_base_model()

    print("Laen adapterid ette...", flush=True)

    first_topic = True
    for topic, adapter_dir in ADAPTER_DIRS.items():
        if not os.path.isdir(adapter_dir):
            raise FileNotFoundError(f"Puudub adapteri kaust: {adapter_dir}")

        print(f"  -> {topic}: {adapter_dir}", flush=True)

        if first_topic:
            # Loob PEFT mudeli esimese adapteriga
            model = PeftModel.from_pretrained(
                model,
                adapter_dir,
                adapter_name=topic,
                is_trainable=False,
            )
            first_topic = False
        else:
            # Lisab sama base mudeli külge järgmised adapterid
            model.load_adapter(
                adapter_dir,
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

    system_msg = SYSTEM_MSGS[topic]
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": question},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    generated_tokens = output[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return answer


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA pole saadaval. Kontrolli PyTorch/CUDA installi.")

    payload = load_router()
    embedder = SentenceTransformer(payload["embedding_model"])
    clf = payload["classifier"]
    threshold = payload.get("threshold", 0.45)

    preload_adapters()

    print("Sisesta küsimus. Väljumiseks kirjuta exit või quit.")

    while True:
        question = input("\nKüsimus> ").strip()
        if not question or question.lower() in {"exit", "quit"}:
            break

        topic, conf, probs, below_threshold = route_question(
            question,
            embedder,
            clf,
            threshold=threshold,
        )

        print(f"Router: {topic}  confidence={conf:.3f}")
        print("Probs:", probs)

        if below_threshold:
            print("Confidence oli madal, kasutan fallback teemana: facts")
            topic = "facts"

        answer = generate_answer(topic, question)
        print("\nVastus:\n", answer)


if __name__ == "__main__":
    main()
