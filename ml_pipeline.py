"""Mental-state classifier: cached pipeline/session loading + analyze_mental_state().

Supersedes the per-call reload in the old LLmText.py prototype.
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import joblib

from config import settings


@lru_cache(maxsize=1)
def _load_pipeline() -> dict[str, Any]:
    return joblib.load(settings.ml_pipeline_path)


@lru_cache(maxsize=1)
def _load_sessions() -> dict[str, Any]:
    with open(settings.sessions_path, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze_mental_state(text: str) -> dict[str, Any]:
    """Classify a piece of text into a mental-state label and return the
    matching canned response (audio_text/tips/yoga) from sessions.json."""
    pipeline = _load_pipeline()
    embedder = pipeline["embedder"]
    clf_model = pipeline["model"]
    classes = list(pipeline["classes"])

    x_emb = embedder.encode([text], batch_size=1, show_progress_bar=False, convert_to_numpy=True)
    pred_result = clf_model.predict(x_emb)[0]
    # The classifier may return either the class label directly (e.g. "Depression")
    # or a numeric index into `classes`, depending on how it was trained.
    if isinstance(pred_result, (int, float)) or (
        hasattr(pred_result, "item") and isinstance(pred_result.item(), (int, float))
    ):
        prediction = classes[int(pred_result)]
    else:
        prediction = str(pred_result)

    session = _load_sessions().get(prediction, {})
    return {
        "prediction": prediction,
        "audio_text": session.get("audio_text", ""),
        "tips": session.get("tips", []),
        "yoga": session.get("yoga", []),
    }


if __name__ == "__main__":
    import sys

    sample = " ".join(sys.argv[1:]) or "I feel hopeless and can't sleep at night."
    print(analyze_mental_state(sample))
