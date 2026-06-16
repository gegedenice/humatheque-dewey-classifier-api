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

# --- Method selection -------------------------------------------------------
# Two interchangeable strategies, chosen per request via `method`:
#   "local"  -> local bi-encoder only (EMBEDDING_MODEL, e.g. multilingual-e5-large)
#   "albert" -> remote bi-encoder retrieval (Albert API, BAAI/bge-m3) to build a
#               candidate pool, then a remote cross-encoder rerank
#               (BAAI/bge-reranker-v2-m3) to reorder it.
METHOD_LOCAL = "local"
METHOD_ALBERT = "albert"
VALID_METHODS = (METHOD_LOCAL, METHOD_ALBERT)
DEFAULT_METHOD = os.getenv("CLASSIFICATION_METHOD", METHOD_LOCAL).lower()

# --- Albert API (https://albert.api.etalab.gouv.fr) -------------------------
ALBERT_API_KEY = os.getenv("ALBERT_API_KEY", "")
ALBERT_BASE_URL = os.getenv("ALBERT_BASE_URL", "https://albert.api.etalab.gouv.fr/v1").rstrip("/")
ALBERT_EMBEDDING_MODEL = os.getenv("ALBERT_EMBEDDING_MODEL", "BAAI/bge-m3")
ALBERT_RERANK_MODEL = os.getenv("ALBERT_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
# bge-m3 is symmetric and uses no query/passage prefixes; override if needed.
ALBERT_QUERY_PREFIX = os.getenv("ALBERT_QUERY_PREFIX", "")
ALBERT_PASSAGE_PREFIX = os.getenv("ALBERT_PASSAGE_PREFIX", "")
ALBERT_TIMEOUT = float(os.getenv("ALBERT_TIMEOUT", "30"))
# Size of the bi-encoder candidate pool handed to the cross-encoder reranker.
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "20"))
# Max inputs per /embeddings request. Albert caps a batch at 64; larger trips 413.
ALBERT_EMBED_BATCH = int(os.getenv("ALBERT_EMBED_BATCH", "64"))


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
    method: str = Field(
        DEFAULT_METHOD,
        description=(
            "Classification strategy. `local`: local bi-encoder embeddings "
            "(multilingual-e5-large) only. `albert`: Albert API bge-m3 retrieval "
            "+ bge-reranker-v2-m3 cross-encoder rerank (requires ALBERT_API_KEY)."
        ),
    )


class ClassifyResponse(BaseModel):
    source: str
    method: str
    model: str
    classification_type: str
    threshold: float
    count: int
    results: list[TextClassification]


class _Index:
    """Bi-encoder index over the taxonomy (and optional examples).

    Subclasses provide the embedding backend (`_encode_passages` / `_encode_queries`)
    and the per-text prefixes. `_retrieve` returns candidate `(row, similarity)`
    pairs sorted by descending bi-encoder similarity; subclasses turn those into the
    final `LabelScore` ranking (the local one directly, the Albert one after a
    cross-encoder rerank).
    """

    query_prefix = ""
    passage_prefix = ""

    def __init__(self, entries: list[dict[str, str]], examples: list[dict[str, str]]):
        import numpy as np

        self.codes = [e["code"] for e in entries]
        self.labels = [e["label"] for e in entries]
        self.code_to_label = {e["code"]: e["label"] for e in entries}
        self.code_to_row = {code: i for i, code in enumerate(self.codes)}
        # Enriched class text: authoritative label + curated keywords. Reused as the
        # bi-encoder passage and as the document handed to the cross-encoder.
        self.descriptions = [f"{e['label']}. {e.get('description', '')}".strip() for e in entries]
        self.class_emb = self._encode_passages([self.passage_prefix + d for d in self.descriptions])

        # Optional catalogued examples for k-NN: one embedding per example, grouped
        # by the Dewey code it was assigned. Empty unless EXAMPLES_PATH is set.
        self.example_rows: dict[str, Any] = {}
        valid = [ex for ex in examples if ex.get("text") and ex.get("code") in self.code_to_row]
        if valid:
            ex_emb = self._encode_passages([self.passage_prefix + ex["text"] for ex in valid])
            by_code: dict[str, list[Any]] = {}
            for ex, vec in zip(valid, ex_emb):
                by_code.setdefault(ex["code"], []).append(vec)
            self.example_rows = {code: np.vstack(vecs) for code, vecs in by_code.items()}

    # --- embedding backend (implemented by subclasses) ---------------------
    def _encode_passages(self, texts: list[str]) -> Any:
        raise NotImplementedError

    def _encode_queries(self, texts: list[str]) -> Any:
        raise NotImplementedError

    def _retrieve(self, text: str, codes: list[str] | None) -> list[tuple[int, float]]:
        query = self._encode_queries([self.query_prefix + text])[0]
        desc_sims = self.class_emb @ query  # cosine, embeddings are normalized

        candidate_rows: Any = range(len(self.codes))
        if codes:
            candidate_rows = [self.code_to_row[c] for c in codes if c in self.code_to_row]
            if not candidate_rows:
                raise HTTPException(status_code=400, detail="No known Dewey codes provided.")

        scored: list[tuple[int, float]] = []
        for row in candidate_rows:
            score = float(desc_sims[row])
            ex = self.example_rows.get(self.codes[row])
            if ex is not None:
                score = max(score, EXAMPLE_WEIGHT * float((ex @ query).max()))
            scored.append((row, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored


class LocalClassifier(_Index):
    """Local bi-encoder only: rank every class by cosine similarity."""

    query_prefix = QUERY_PREFIX
    passage_prefix = PASSAGE_PREFIX

    def __init__(self, model: Any, entries: list[dict[str, str]], examples: list[dict[str, str]]):
        self.model = model
        super().__init__(entries, examples)

    def _encode_passages(self, texts: list[str]) -> Any:
        return self.model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )

    _encode_queries = _encode_passages

    def rank(
        self, text: str, codes: list[str] | None, threshold: float, top_k: int
    ) -> list[LabelScore]:
        ranked = self._retrieve(text, codes)
        scored = [
            LabelScore(dewey=self.codes[row], label=self.labels[row], score=round(score, 4))
            for row, score in ranked
        ]
        scored = [s for s in scored if s.score >= threshold]
        return scored[:top_k]


class AlbertClassifier(_Index):
    """Albert API: bi-encoder (bge-m3) retrieval, then cross-encoder rerank.

    The bi-encoder narrows the taxonomy to a `RERANK_CANDIDATES`-sized pool; the
    cross-encoder (bge-reranker-v2-m3) then scores each candidate's enriched
    description against the query. Returned scores are reranker relevance scores,
    not cosine similarities.
    """

    query_prefix = ALBERT_QUERY_PREFIX
    passage_prefix = ALBERT_PASSAGE_PREFIX

    def _embed(self, texts: list[str]) -> Any:
        import numpy as np

        vectors: list[list[float]] = []
        for start in range(0, len(texts), ALBERT_EMBED_BATCH):
            batch = texts[start : start + ALBERT_EMBED_BATCH]
            payload = _albert_post("/embeddings", {"model": ALBERT_EMBEDDING_MODEL, "input": batch})
            data = sorted(payload["data"], key=lambda d: d["index"])
            vectors.extend(d["embedding"] for d in data)
        arr = np.asarray(vectors, dtype="float32")
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms  # L2-normalized, so the dot product is cosine similarity

    _encode_passages = _embed
    _encode_queries = _embed

    def rank(
        self, text: str, codes: list[str] | None, threshold: float, top_k: int
    ) -> list[LabelScore]:
        ranked = self._retrieve(text, codes)
        pool = ranked[: max(top_k, RERANK_CANDIDATES)]
        if not pool:
            return []

        documents = [self.descriptions[row] for row, _ in pool]
        payload = _albert_post(
            "/rerank",
            {"model": ALBERT_RERANK_MODEL, "query": text, "documents": documents},
        )
        results = sorted(
            payload["results"], key=lambda r: r["relevance_score"], reverse=True
        )

        scored = []
        for r in results:
            row = pool[r["index"]][0]
            score = round(float(r["relevance_score"]), 4)
            scored.append(LabelScore(dewey=self.codes[row], label=self.labels[row], score=score))
        scored = [s for s in scored if s.score >= threshold]
        return scored[:top_k]


def _albert_post(path: str, body: dict[str, Any]) -> dict[str, Any]:
    """POST to the Albert API and return the parsed JSON, mapping failures to HTTP errors."""
    import httpx

    if not ALBERT_API_KEY:
        raise HTTPException(
            status_code=503, detail="ALBERT_API_KEY is not configured for the 'albert' method."
        )
    headers = {"Authorization": f"Bearer {ALBERT_API_KEY}"}
    try:
        resp = httpx.post(f"{ALBERT_BASE_URL}{path}", json=body, headers=headers, timeout=ALBERT_TIMEOUT)
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Albert API error ({exc.response.status_code}): {exc.response.text[:200]}",
        )
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Albert API request failed: {exc}")
    return resp.json()


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


@lru_cache(maxsize=len(VALID_METHODS))
def get_classifier(method: str) -> _Index:
    """Build the taxonomy index for `method` once per process (cached per method)."""
    if method == METHOD_ALBERT:
        if not ALBERT_API_KEY:
            raise HTTPException(
                status_code=503, detail="ALBERT_API_KEY is not configured for the 'albert' method."
            )
        return AlbertClassifier(load_taxonomy(), load_examples())

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL, device=EMBEDDING_DEVICE, token=HF_TOKEN)
    return LocalClassifier(model, load_taxonomy(), load_examples())


def classify(payload: ClassifyRequest) -> dict[str, Any]:
    texts = [payload.text] if isinstance(payload.text, str) else list(payload.text)
    if not texts or all(not text.strip() for text in texts):
        raise HTTPException(status_code=400, detail="Provide non-empty text.")

    method = (payload.method or DEFAULT_METHOD).lower()
    if method not in VALID_METHODS:
        raise HTTPException(
            status_code=400, detail=f"Unknown method {method!r}; expected one of {list(VALID_METHODS)}."
        )

    top_k = 1 if payload.classification_type == "single-label" else (payload.top_k or DEFAULT_TOP_K)
    classifier = get_classifier(method)

    results = [
        TextClassification(
            text=text,
            classes=classifier.rank(text, payload.codes, payload.threshold, top_k),
        )
        for text in texts
    ]

    if method == METHOD_ALBERT:
        source = "albert_rerank_classification"
        model_name = f"{ALBERT_EMBEDDING_MODEL} + {ALBERT_RERANK_MODEL}"
    else:
        source = "embedding_classification"
        model_name = EMBEDDING_MODEL

    return {
        "source": source,
        "method": method,
        "model": model_name,
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
