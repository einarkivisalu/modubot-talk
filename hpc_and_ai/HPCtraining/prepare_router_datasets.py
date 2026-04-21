import json
import random
from pathlib import Path

TOPIC_FILES = {
    "facts": "training_materials/huvitavad_faktid.json",
    "kasitoo": "training_materials/kasitoo.json",
    #"kunstiajalugu": "training_materials/kunsti_ajalugu.json",
    "luuletused": "training_materials/luuletused.json",
    "muistendid": "training_materials/muistendit.json",
    "tahtpaevad": "training_materials/tahtpaevad.json",
    "moistatused": "training_materials/moistatused.json",
}

FOLLOW_UP_TEMPLATES = [
    "Räägi veel sellest.",
    "Selgita seda täpsemalt.",
    "Kas sa saad seda edasi selgitada?",
    "Jätka palun.",
]

OUT_BASELINE = Path("router_data/router_baseline.jsonl")
OUT_CONTEXT = Path("router_data/router_context.jsonl")


def load_questions(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    if not isinstance(rows, list):
        raise ValueError(f"{path} peab olema list.")

    questions = []
    for row in rows:
        q = (row.get("question") or "").strip()
        if q:
            questions.append(q)
    return questions


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    baseline_rows = []
    context_rows = []

    topics = list(TOPIC_FILES.keys())

    for topic, file_path in TOPIC_FILES.items():
        questions = load_questions(file_path)

        # Baseline: ainult küsimus -> teema
        for q in questions:
            baseline_rows.append(
                {
                    "text": q,
                    "label": topic,
                }
            )

        # Context: päris küsimus + lihtne kontekst
        for q in questions:
            context_rows.append(
                {
                    "text": q,
                    "label": topic,
                    "last_topic": topic,
                    "turn_index": 1,
                    "is_follow_up": 0,
                }
            )

        # Context: sama follow-up lause eri teemade all
        # See aitab mudelil õppida, et kontekst võib otsust muuta.
        for prev_topic in topics:
            for template in FOLLOW_UP_TEMPLATES:
                context_rows.append(
                    {
                        "text": template,
                        "label": topic,
                        "last_topic": prev_topic,
                        "turn_index": 2,
                        "is_follow_up": 1,
                    }
                )

    random.shuffle(baseline_rows)
    random.shuffle(context_rows)

    write_jsonl(OUT_BASELINE, baseline_rows)
    write_jsonl(OUT_CONTEXT, context_rows)

    print(f"Kirjutatud: {OUT_BASELINE}")
    print(f"Kirjutatud: {OUT_CONTEXT}")


if __name__ == "__main__":
    main()
