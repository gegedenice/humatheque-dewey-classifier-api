# Humatheque Classification API

Local FastAPI service that assigns a **Dewey class** to short academic text
(titles, occasionally abstracts) by **semantic similarity** — fully offline, with
open-weights models and no paid API.

The service is designed for a cataloging pipeline where a piece of text needs a
discipline / Dewey assignment. A client posts the text; the API embeds it with a
local multilingual sentence-embedding model, ranks it against the Dewey taxonomy,
and returns the best-matching classes with a similarity score. Each returned
class carries `dewey` + `label` + `score`.

## How it works

The classes are a fixed, authoritative Dewey taxonomy stored in `taxonomy.json`.
Each entry has three fields:

```json
{"code": "980", "label": "Histoire générale de l'Amérique du Sud",
 "description": "Histoire de l'Amérique du Sud et latine, Argentine, Brésil, Buenos Aires, indépendances sud-américaines"}
```

- `code` + `label` are **authoritative** — they are exactly what the API returns.
- `description` is **internal**: an enriched set of keywords (place names, eras,
  fields, synonyms) used only to build the class embedding.

Why the enrichment matters: incoming titles are terse and very specific (e.g. a
thesis subject), and must roll up to a *broader* category. A bare label string
cannot connect "Buenos Aires, 1829" to "Amérique du Sud" — the discriminating
vocabulary in `description` is what makes that mapping work. **Improving accuracy
is done by editing `taxonomy.json`, not by tuning the model**, and a cataloger
can do it without touching Python.

### Classification logic

`POST /classify`:

1. Embed each Dewey class once at startup as `"{label}. {description}"` (with the
   model's passage prefix). The index is built once per process and cached.
2. Embed the incoming text (with the query prefix).
3. Rank classes by cosine similarity, drop anything below `threshold`, sort, and
   return the top `top_k`.

`multi-label` (default) returns up to `top_k` classes; `single-label` returns the
single best class.

### Model

The default model is
[`intfloat/multilingual-e5-large`](https://huggingface.co/intfloat/multilingual-e5-large),
served via `sentence-transformers`. It is multilingual (strong on French), runs
on CPU, and is fully local. Swap it with `EMBEDDING_MODEL` (e.g. `BAAI/bge-m3`,
`intfloat/multilingual-e5-base` for lower latency).

e5/bge models use asymmetric prefixes — the search text is a `query: ` and the
class descriptions are `passage: `. These are configurable and **must keep their
trailing space** (hence the quoting in `.example.env`).

### Improving over time (k-NN)

The index can also match against previously catalogued examples. Point
`EXAMPLES_PATH` at a JSON file of confirmed assignments:

```json
[
  {"text": "Étude des algorithmes d'apprentissage automatique", "code": "004"},
  {"text": "Histoire politique de Buenos Aires au XIXe siècle", "code": "980"}
]
```

Each example is embedded per class and blended with the description similarity as
`final = max(description_similarity, EMBEDDING_EXAMPLE_WEIGHT × best_example_similarity)`,
so confirmed assignments can only improve results. Leave `EXAMPLES_PATH` empty to
disable.

## Endpoints

Interactive documentation is available at `/docs`.

### `GET /health`

Container health check.

```json
{"ok": true}
```

### `POST /classify`

Request body:

```json
{
  "text": "Handbook on large language models and embeddings models.",
  "codes": null,
  "threshold": 0.0,
  "classification_type": "multi-label",
  "top_k": 5
}
```

| Field | Required | Description |
|---|---:|---|
| `text` | yes | A string, or a list of strings for batch classification |
| `codes` | no | Optional subset of Dewey codes to restrict candidates to (e.g. `["004","510"]`); defaults to the full taxonomy. Unknown codes are ignored |
| `threshold` | no | Minimum cosine similarity to return a class, default `0.0` |
| `classification_type` | no | `multi-label` (default, up to `top_k`) or `single-label` (best one) |
| `top_k` | no | Cap on returned classes per text; default from `CLASSIFICATION_TOP_K` (`5`) |

Example:

```bash
curl -s localhost:8000/classify -H 'Content-Type: application/json' -d '{
  "text": "Handbook on large language models and embeddings models.",
  "top_k": 3
}'
```

Response shape:

```jsonc
{
  "source": "embedding_classification",
  "model": "intfloat/multilingual-e5-large",
  "classification_type": "multi-label",
  "threshold": 0.0,
  "count": 1,
  "results": [
    {
      "text": "Handbook on large language models and embeddings models.",
      "classes": [
        {"dewey": "004", "label": "Informatique", "score": 0.88},
        {"dewey": "410", "label": "Linguistique générale", "score": 0.84},
        {"dewey": "510", "label": "Mathématiques", "score": 0.81}
      ]
    }
  ]
}
```

Batch classification posts a list of texts and returns one entry per text:

```json
{"text": ["Premier texte à classer.", "Second texte à classer."], "top_k": 3}
```

> **Note on scores:** these are cosine similarities, not calibrated
> probabilities. With e5-style models they cluster high (≈0.7–0.9) even for weak
> matches, so treat them as a *ranking*. `threshold` defaults to `0.0`; rely on
> `top_k` and human-in-the-loop confirmation rather than a hard cutoff.

## Authentication

Authentication is optional. Set `CLASSIFICATION_API_KEY` (or `API_KEY`); clients
must then send:

```text
X-API-Key: <key>
```

If the key is empty, the endpoint is public.

## Environment variables

Copy `.example.env` to `.env` and adjust values.

| Variable | Default | Description |
|---|---|---|
| `PORT` | `8000` | HTTP server port |
| `CLASSIFICATION_API_KEY` | empty | Optional API key (`API_KEY` is also honored) |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-large` | Sentence-embedding model identifier |
| `EMBEDDING_DEVICE` | `cpu` | Inference device (`cpu`, `cuda:0`, ...) |
| `HF_TOKEN` | empty | Optional Hugging Face token for downloading the model (`HUGGING_FACE_HUB_TOKEN` is also honored); needed for gated/private models |
| `EMBEDDING_QUERY_PREFIX` | `"query: "` | Prefix for the search text (model-specific; keep trailing space) |
| `EMBEDDING_PASSAGE_PREFIX` | `"passage: "` | Prefix for class descriptions / examples (keep trailing space) |
| `TAXONOMY_PATH` | `taxonomy.json` | Path to the authoritative Dewey taxonomy |
| `EXAMPLES_PATH` | empty | Optional JSON of catalogued `{text, code}` examples for k-NN |
| `EMBEDDING_EXAMPLE_WEIGHT` | `1.0` | Weight on the best example similarity when blending |
| `CLASSIFICATION_TOP_K` | `5` | Default cap on returned classes per text |
| `CLASSIFICATION_THRESHOLD` | `0.0` | Default minimum cosine similarity |
| `CLASSIFICATION_TYPE` | `multi-label` | Default classification type |

## Local run

```bash
cp .example.env .env
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The first request downloads the embedding model from Hugging Face and builds the
taxonomy index; subsequent requests reuse the loaded model.

## Docker

```bash
docker build -t humatheque-classification-api .
docker run --env-file .env -p 8000:8000 humatheque-classification-api
```

The Hugging Face cache lives under `/app/.cache/huggingface`; mount a volume
there to avoid re-downloading the model on each container start.

## Operational notes

- The model and taxonomy index are loaded once and cached for the process
  lifetime. **Restart the service** to pick up changes to `taxonomy.json`,
  `EXAMPLES_PATH`, or the model.
- Embedding is CPU-bound; it runs in a threadpool so it does not block the event
  loop, but a single worker processes one request at a time.
- For throughput, prefer batch requests (a list of texts).
- Accuracy is driven by the `description` keywords in `taxonomy.json` and by the
  optional catalogued examples — not by model parameters.
