# router.py

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer


class TopicRouter:
    def __init__(self, model_path: str):
        payload = joblib.load(model_path)

        self.embedding_model_name = payload["embedding_model_name"]
        self.use_context = payload["use_context"]
        self.classifier = payload["classifier"]
        self.topic_encoder = payload["topic_encoder"]
        self.num_scaler = payload["num_scaler"]

        self.embedder = SentenceTransformer(self.embedding_model_name)

    def _build_row(self, text: str, last_topic: str = "unknown", turn_index: int = 1, is_follow_up: int = 0):
        return {
            "text": text,
            "last_topic": last_topic,
            "turn_index": turn_index,
            "is_follow_up": is_follow_up,
        }

    def _embed_texts(self, texts):
        return self.embedder.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def _numeric_matrix(self, rows):
        return np.array(
            [
                [float(r.get("turn_index", 1)), float(r.get("is_follow_up", 0))]
                for r in rows
            ],
            dtype=np.float32,
        )

    def _transform(self, rows):
        texts = [r["text"] for r in rows]
        x_text = self._embed_texts(texts)

        if not self.use_context:
            return x_text

        last_topics = np.array([[r.get("last_topic", "unknown")] for r in rows], dtype=object)
        x_topic = self.topic_encoder.transform(last_topics)
        x_num = self.num_scaler.transform(self._numeric_matrix(rows))

        return np.hstack([x_text, x_topic, x_num])

    def predict(self, text: str, last_topic: str = "unknown", turn_index: int = 1, is_follow_up: int = 0, threshold: float = 0.45):
        row = self._build_row(
            text=text,
            last_topic=last_topic,
            turn_index=turn_index,
            is_follow_up=is_follow_up,
        )

        x = self._transform([row])
        proba = self.classifier.predict_proba(x)[0]
        classes = self.classifier.classes_

        best_idx = int(np.argmax(proba))
        best_topic = classes[best_idx]
        confidence = float(proba[best_idx])

        if confidence < threshold:
            return "fallback", confidence

        return best_topic, confidence

    def predict_with_probs(self, text: str, last_topic: str = "unknown", turn_index: int = 1, is_follow_up: int = 0):
        row = self._build_row(
            text=text,
            last_topic=last_topic,
            turn_index=turn_index,
            is_follow_up=is_follow_up,
        )

        x = self._transform([row])
        proba = self.classifier.predict_proba(x)[0]
        classes = self.classifier.classes_

        results = []
        for i, cls in enumerate(classes):
            results.append((cls, float(proba[i])))

        results.sort(key=lambda item: item[1], reverse=True)
        return results
