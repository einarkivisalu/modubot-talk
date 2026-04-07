import json
import os
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

TOPIC_FILES = {
    "facts": "training_materials/huvitavad_faktid.json",
    "kasitoo": "training_materials/kasitoo.json",
    "kunstiajalugu": "training_materials/luuletused.json",
}

OUT_DIR = "router_artifacts"
ROUTER_PATH = os.path.join(OUT_DIR, "topic_router.joblib")


def load_router_examples(topic_files):
    texts = []
    labels = []

    for label, path in topic_files.items():
        if not os.path.isfile(path):
            raise RuntimeError(f"Missing file: {path}")

        with open(path, "r", encoding="utf-8") as f:
            rows = json.load(f)

        if not isinstance(rows, list):
            raise RuntimeError(f"Expected list in {path}")

        for row in rows:
            q = row.get("question", "").strip()
            if q:
                texts.append(q)
                labels.append(label)

    return texts, labels


def main():
    texts, labels = load_router_examples(TOPIC_FILES)

    if len(texts) < 10:
        raise RuntimeError("Router training data is too small. Add more questions per topic.")

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    embedder = SentenceTransformer(EMBEDDING_MODEL)

    X_train = embedder.encode(X_train_text, normalize_embeddings=True, show_progress_bar=True)
    X_test = embedder.encode(X_test_text, normalize_embeddings=True, show_progress_bar=True)

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    )
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print("Accuracy:", acc)
    print(classification_report(y_test, preds))

    os.makedirs(OUT_DIR, exist_ok=True)
    payload = {
        "embedding_model": EMBEDDING_MODEL,
        "classifier": clf,
        "topics": list(TOPIC_FILES.keys()),
        "topic_files": TOPIC_FILES,
        "threshold": 0.45,
    }
    joblib.dump(payload, ROUTER_PATH)
    print(f"Saved router to {ROUTER_PATH}")


if __name__ == "__main__":
    main()
