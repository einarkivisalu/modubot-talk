// käsk muutujateta treenimiseks
// python train_router.py --input router_data/router_baseline.jsonl --output router_artifacts/router_baseline.joblib

// käsk muutujatega treenimiseks
// python train_router.py --input router_data/router_context.jsonl --output router_artifacts/router_context.joblib

import argparse
import json
import joblib
import numpy as np
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class TopicRouter:
    def __init__(self, embedding_model_name: str, use_context: bool):
        self.embedding_model_name = embedding_model_name
        self.use_context = use_context
        self.embedder = SentenceTransformer(embedding_model_name)
        self.classifier = LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="lbfgs",
        )

        self.topic_encoder = None
        self.num_scaler = None

        if self.use_context:
            self.topic_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            self.num_scaler = StandardScaler()

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        return self.embedder.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def _numeric_matrix(self, rows: list[dict]) -> np.ndarray:
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

    def fit(self, train_rows: list[dict]):
        texts = [r["text"] for r in train_rows]
        x_text = self._embed_texts(texts)

        if self.use_context:
            last_topics = np.array([[r.get("last_topic", "unknown")] for r in train_rows], dtype=object)
            x_topic = self.topic_encoder.fit_transform(last_topics)
            x_num = self.num_scaler.fit_transform(self._numeric_matrix(train_rows))
            x = np.hstack([x_text, x_topic, x_num])
        else:
            x = x_text

        y = [r["label"] for r in train_rows]
        self.classifier.fit(x, y)
        return self

    def transform(self, rows: list[dict]) -> np.ndarray:
        texts = [r["text"] for r in rows]
        x_text = self._embed_texts(texts)

        if self.use_context:
            last_topics = np.array([[r.get("last_topic", "unknown")] for r in rows], dtype=object)
            x_topic = self.topic_encoder.transform(last_topics)
            x_num = self.num_scaler.transform(self._numeric_matrix(rows))
            return np.hstack([x_text, x_topic, x_num])

        return x_text

    def predict(self, rows: list[dict]) -> np.ndarray:
        x = self.transform(rows)
        return self.classifier.predict(x)

    def predict_proba(self, rows: list[dict]) -> np.ndarray:
        x = self.transform(rows)
        return self.classifier.predict_proba(x)

    def save(self, path: str):
        payload = {
            "embedding_model_name": self.embedding_model_name,
            "use_context": self.use_context,
            "classifier": self.classifier,
            "topic_encoder": self.topic_encoder,
            "num_scaler": self.num_scaler,
        }
        joblib.dump(payload, path)

    @staticmethod
    def load(path: str):
        payload = joblib.load(path)
        router = TopicRouter(
            embedding_model_name=payload["embedding_model_name"],
            use_context=payload["use_context"],
        )
        router.classifier = payload["classifier"]
        router.topic_encoder = payload["topic_encoder"]
        router.num_scaler = payload["num_scaler"]
        return router


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="router_artifacts/router.joblib")
    parser.add_argument("--embedding_model", type=str, default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    if len(rows) < 20:
        raise RuntimeError("Routeri andmestik on liiga väike. Lisa rohkem näiteid.")

    use_context = any("last_topic" in r for r in rows)

    labels = [r["label"] for r in rows]
    train_rows, test_rows = train_test_split(
        rows,
        test_size=args.test_size,
        random_state=42,
        stratify=labels,
    )

    router = TopicRouter(args.embedding_model, use_context=use_context)
    router.fit(train_rows)

    preds = router.predict(test_rows)
    y_true = [r["label"] for r in test_rows]

    acc = accuracy_score(y_true, preds)
    print("Accuracy:", acc)
    print("\nClassification report:\n", classification_report(y_true, preds))
    print("\nConfusion matrix:\n", confusion_matrix(y_true, preds))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    router.save(args.output)
    print(f"Salvestatud: {args.output}")


if __name__ == "__main__":
    main()
