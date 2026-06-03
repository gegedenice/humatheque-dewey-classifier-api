"""FastAPI service for local, zero-shot Dewey classification by embedding retrieval.

The incoming text is embedded with a local multilingual sentence-embedding model
and ranked by cosine similarity against the Dewey taxonomy. Each class is
represented by an *enriched description* (authoritative label + curated keywords)
so that very specific titles map onto the correct broad category -- which a bare
label string cannot do. Optionally, previously catalogued `{text -> code}`
examples are matched as well (k-NN), so the system improves as the catalogue
grows. Only the authoritative `code` + `label` are ever returned.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.concurrency import run_in_threadpool
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
# Optional Hugging Face access token, used to download the embedding model
# (required for gated/private models, raises anonymous rate limits otherwise).
HF_TOKEN = os.getenv("HF_TOKEN", os.getenv("HUGGING_FACE_HUB_TOKEN", "")) or None
# e5 / bge models are trained with asymmetric prefixes; the search text is a
# "query" and the class descriptions / examples are "passages". Override (e.g. to
# empty strings) for models that don't use prefixes.
QUERY_PREFIX = os.getenv("EMBEDDING_QUERY_PREFIX", "query: ")
PASSAGE_PREFIX = os.getenv("EMBEDDING_PASSAGE_PREFIX", "passage: ")

TAXONOMY_PATH = os.getenv("TAXONOMY_PATH", str(Path(__file__).parent / "taxonomy.json"))
# Optional JSON file of catalogued examples for k-NN: [{"text": ..., "code": ...}].
EXAMPLES_PATH = os.getenv("EXAMPLES_PATH", "")
# Weight applied to the best matching example's similarity when blending it with
# the class-description similarity (final = max(desc_sim, weight * example_sim)).
EXAMPLE_WEIGHT = float(os.getenv("EMBEDDING_EXAMPLE_WEIGHT", "1.0"))

DEFAULT_TOP_K = int(os.getenv("CLASSIFICATION_TOP_K", "5"))
DEFAULT_THRESHOLD = float(os.getenv("CLASSIFICATION_THRESHOLD", "0.0"))
DEFAULT_CLASSIFICATION_TYPE = os.getenv("CLASSIFICATION_TYPE", "multi-label")
API_KEY = os.getenv("CLASSIFICATION_API_KEY", os.getenv("API_KEY", ""))


app = FastAPI(
    title="Humatheque Classification API",
    version="0.2.0",
    description=(
        "Local Dewey classification of academic text by semantic similarity. The "
        "text is embedded with a multilingual sentence-embedding model and ranked "
        "against the Dewey taxonomy, where each class is described by its "
        "authoritative label plus curated keywords. Returns the matching classes "
        "as `dewey` + `label` + `score` (cosine similarity)."
    ),
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(api_key: str | None = Security(api_key_header)) -> None:
    if API_KEY and api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


class LabelScore(BaseModel):
    dewey: str | None = Field(None, description="Dewey code of the matched class.")
    label: str = Field(..., description="Authoritative Dewey discipline label.")
    score: float = Field(..., description="Cosine similarity of the text to the class.")


class TextClassification(BaseModel):
    text: str
    classes: list[LabelScore]


class ClassifyRequest(BaseModel):
    text: str | list[str] = Field(
        ...,
        description="Text to classify, or a list of texts for batch classification.",
    )
    codes: list[str] | None = Field(
        None,
        description=(
            "Optional subset of Dewey codes to restrict the candidate classes to "
            "(e.g. ['004', '510']). Defaults to the full taxonomy. Unknown codes "
            "are ignored."
        ),
    )
    threshold: float = Field(
        DEFAULT_THRESHOLD,
        ge=-1.0,
        le=1.0,
        description="Minimum cosine similarity for a class to be returned.",
    )
    classification_type: str = Field(
        DEFAULT_CLASSIFICATION_TYPE,
        description="`multi-label` returns up to top_k classes; `single-label` returns the best one.",
    )
    top_k: int | None = Field(
        None,
        ge=1,
        description="Cap on the number of returned classes per text (default from env).",
    )


class ClassifyResponse(BaseModel):
    source: str
    model: str
    classification_type: str
    threshold: float
    count: int
    results: list[TextClassification]


class Classifier:
    """Embeds the taxonomy (and optional examples) once and ranks texts against it."""

    def __init__(self, model: Any, entries: list[dict[str, str]], examples: list[dict[str, str]]):
        import numpy as np

        self.model = model
        self.codes = [e["code"] for e in entries]
        self.labels = [e["label"] for e in entries]
        self.code_to_label = {e["code"]: e["label"] for e in entries}
        self.code_to_row = {code: i for i, code in enumerate(self.codes)}

        passages = [f"{PASSAGE_PREFIX}{e['label']}. {e.get('description', '')}".strip() for e in entries]
        self.class_emb = self._encode(passages)

        # Optional catalogued examples for k-NN: one embedding per example, grouped
        # by the Dewey code it was assigned. Empty unless EXAMPLES_PATH is set.
        self.example_rows: dict[str, Any] = {}
        valid = [ex for ex in examples if ex.get("text") and ex.get("code") in self.code_to_row]
        if valid:
            ex_emb = self._encode([f"{PASSAGE_PREFIX}{ex['text']}" for ex in valid])
            by_code: dict[str, list[Any]] = {}
            for ex, vec in zip(valid, ex_emb):
                by_code.setdefault(ex["code"], []).append(vec)
            self.example_rows = {code: np.vstack(vecs) for code, vecs in by_code.items()}

    def _encode(self, texts: list[str]) -> Any:
        return self.model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )

    def rank(
        self, text: str, codes: list[str] | None, threshold: float, top_k: int
    ) -> list[LabelScore]:
        import numpy as np

        query = self._encode([f"{QUERY_PREFIX}{text}"])[0]
        desc_sims = self.class_emb @ query  # cosine, embeddings are normalized

        candidate_rows = range(len(self.codes))
        if codes:
            candidate_rows = [self.code_to_row[c] for c in codes if c in self.code_to_row]
            if not candidate_rows:
                raise HTTPException(status_code=400, detail="No known Dewey codes provided.")

        scored: list[LabelScore] = []
        for row in candidate_rows:
            code = self.codes[row]
            score = float(desc_sims[row])
            ex = self.example_rows.get(code)
            if ex is not None:
                score = max(score, EXAMPLE_WEIGHT * float((ex @ query).max()))
            scored.append(LabelScore(dewey=code, label=self.labels[row], score=round(score, 4)))

        scored.sort(key=lambda item: item.score, reverse=True)
        scored = [s for s in scored if s.score >= threshold]
        return scored[:top_k]


def load_taxonomy() -> list[dict[str, str]]:
    with open(TAXONOMY_PATH, encoding="utf-8") as fh:
        entries = json.load(fh)
    if not entries:
        raise RuntimeError(f"Taxonomy at {TAXONOMY_PATH!r} is empty.")
    return entries


def load_examples() -> list[dict[str, str]]:
    if not EXAMPLES_PATH or not Path(EXAMPLES_PATH).exists():
        return []
    with open(EXAMPLES_PATH, encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def get_classifier() -> Classifier:
    """Load the embedding model and build the taxonomy index once per process."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL, device=EMBEDDING_DEVICE, token=HF_TOKEN)
    return Classifier(model, load_taxonomy(), load_examples())


def classify(payload: ClassifyRequest) -> dict[str, Any]:
    texts = [payload.text] if isinstance(payload.text, str) else list(payload.text)
    if not texts or all(not text.strip() for text in texts):
        raise HTTPException(status_code=400, detail="Provide non-empty text.")

    top_k = 1 if payload.classification_type == "single-label" else (payload.top_k or DEFAULT_TOP_K)
    classifier = get_classifier()

    results = [
        TextClassification(
            text=text,
            classes=classifier.rank(text, payload.codes, payload.threshold, top_k),
        )
        for text in texts
    ]

    return {
        "source": "embedding_classification",
        "model": EMBEDDING_MODEL,
        "classification_type": payload.classification_type,
        "threshold": payload.threshold,
        "count": len(results),
        "results": results,
    }


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "humatheque-classification-api",
        "version": app.version,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True}


@app.post("/classify", response_model=ClassifyResponse)
async def classify_endpoint(
    payload: ClassifyRequest,
    _: None = Depends(require_api_key),
) -> dict[str, Any]:
    return await run_in_threadpool(classify, payload)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8002)
