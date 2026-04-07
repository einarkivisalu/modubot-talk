import os
import joblib
import torch
from functools import lru_cache

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


@lru_cache(maxsize=None)
def load_adapter(topic):
    if topic not in ADAPTER_DIRS:
        raise ValueError(f"Tundmatu teema: {topic}")

    adapter_dir = ADAPTER_DIRS[topic]

    if not os.path.isdir(adapter_dir):
        raise FileNotFoundError(f"Puudub adapteri kaust: {adapter_dir}")

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA pole saadaval.")

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to("cuda")

    base_model.config.use_cache = True
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()

    return tokenizer, model


def generate_answer(topic, question):
    tokenizer, model = load_adapter(topic)
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

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    # võta ainult uus genereeritud osa
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
