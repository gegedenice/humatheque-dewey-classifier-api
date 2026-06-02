# Humatheque Classification API

FastAPI service for zero-shot classification of academic text against a Dewey
taxonomy, using the GLiClass `gliclass-modern-large-v3.0` model.

The service is designed for a cataloging pipeline where text (a title, abstract,
or extracted document content) needs a discipline / Dewey assignment. A client
posts the text and a candidate taxonomy, and the API returns the matching
classes with their model confidence scores.

The taxonomy is supplied as a list of `{dewey_code: discipline_label}` entries.
Only the discipline label is sent to the model (GLiClass classifies on the label
text), and the response re-attaches the Dewey code to every scored class, so each
returned class carries `dewey` + `label` + `score`.

## Model

The service uses the GLiClass library and the
[`knowledgator/gliclass-modern-large-v3.0`](https://huggingface.co/knowledgator/gliclass-modern-large-v3.0)
model, served through `ZeroShotClassificationPipeline`:

```python
from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from transformers import AutoTokenizer
import torch

model = GLiClassModel.from_pretrained("knowledgator/gliclass-modern-large-v3.0", dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained("knowledgator/gliclass-modern-large-v3.0")
pipeline = ZeroShotClassificationPipeline(model, tokenizer, classification_type="multi-label", device="cpu")
```

The pipeline is built once per `classification_type` and cached for the lifetime
of the process. The first request pays the model-load cost; subsequent requests
reuse the loaded model. Inference runs in a threadpool so the event loop is not
blocked.

GLiClass natively supports both single-text and batch classification, so this
service accepts either a single `text` string or a list of texts in one request.

## Classification logic

`POST /classify`:

1. Normalize `text` to a list (single string or batch).
2. Parse `labels` (`[{code: label}, ...]`) into:
   - the ordered list of discipline labels passed to the model
   - a `label -> [dewey codes]` map used to re-attach codes in the response
3. Run the GLiClass pipeline with the requested `threshold`.
4. For each text, return classes above the threshold as `dewey` + `label` +
   `score`, sorted by descending score (optionally capped by `top_k`).

`multi-label` (default) returns every label whose score passes the threshold.
`single-label` returns the single best class.

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
  "text": "Étude des algorithmes d'apprentissage automatique appliqués au traitement du langage.",
  "labels": [
    {"000": "Informatique, information, généralités"},
    {"004": "Informatique"},
    {"020": "Bibliothéconomie et sciences de l'information"},
    {"100": "Les divers systèmes et écoles philosophiques"},
    {"200": "Religion"}
  ],
  "threshold": 0.5,
  "classification_type": "multi-label",
  "top_k": null
}
```

| Field | Required | Description |
|---|---:|---|
| `text` | yes | A string, or a list of strings for batch classification |
| `labels` | no | List of `{dewey_code: discipline_label}` entries; defaults to the Dewey divisions (main classes and hundred-level subdivisions) |
| `threshold` | no | Minimum score to return a class, default `0.5` |
| `classification_type` | no | `multi-label` (default) or `single-label` |
| `top_k` | no | Optional cap on returned classes per text |

`labels` only sends the discipline label to the model; the Dewey code is mapped
back onto each result. If several codes share the same label text, every code is
returned for that label.

Example:

```bash
curl -X POST "http://localhost:8000/classify" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${API_KEY}" \
  -d '{
    "text": "Étude des algorithmes dapprentissage automatique appliqués au traitement du langage.",
    "labels": [
      {"004": "Informatique"},
      {"100": "Les divers systèmes et écoles philosophiques"},
      {"200": "Religion"}
    ],
    "threshold": 0.5
  }'
```

Response shape:

```jsonc
{
  "source": "gliclass_classification",
  "model": "knowledgator/gliclass-modern-large-v3.0",
  "classification_type": "multi-label",
  "threshold": 0.5,
  "count": 1,
  "results": [
    {
      "text": "Étude des algorithmes d'apprentissage automatique...",
      "classes": [
        {"dewey": "200", "label": "Religion", "score": 0.92},
        {"dewey": "004", "label": "Informatique", "score": 0.916},
        {"dewey": "100", "label": "Les divers systèmes et écoles philosophiques", "score": 0.904}
      ]
    }
  ]
}
```

Batch classification posts a list of texts and returns one entry per text in
`results`:

```json
{
  "text": ["Premier texte à classer.", "Second texte à classer."],
  "labels": [{"004": "Informatique"}, {"200": "Religion"}]
}
```

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
| `GLICLASS_MODEL` | `knowledgator/gliclass-modern-large-v3.0` | GLiClass model identifier |
| `GLICLASS_DEVICE` | `cpu` | Inference device (`cpu`, `cuda:0`, ...) |
| `GLICLASS_DTYPE` | `float32` | Model dtype (`float32`, `float16`, `bfloat16`) |
| `GLICLASS_CLASSIFICATION_TYPE` | `multi-label` | Default classification type |
| `GLICLASS_THRESHOLD` | `0.5` | Default score threshold |
| `GLICLASS_MAX_LENGTH` | `1024` | Maximum token length per text |

## Local run

```bash
cp .example.env .env
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The first request downloads the model from Hugging Face and loads it into
memory, so it is slower than later requests.

## Docker

```bash
docker build -t humatheque-classification-api .
docker run --env-file .env -p 8000:8000 humatheque-classification-api
```

The Docker image follows the same deployment style as the other Humatheque
services: Python slim image, `requirements.txt`, non-root user, and
`uvicorn app:app`. The Hugging Face cache lives under `/app/.cache/huggingface`;
mount a volume there to avoid re-downloading the model on each container start.

## Operational notes

- The model is loaded lazily on the first request and cached per
  `classification_type` for the process lifetime.
- Inference is CPU-bound; it runs in a threadpool so it does not block the event
  loop, but a single worker processes one batch at a time.
- For throughput, prefer batch requests (a list of texts) over many single-text
  requests, since GLiClass batches them in one forward pass.
- `gliclass-modern-large-v3.0` is the large variant; for lower-latency serving
  consider a smaller GLiClass model via `GLICLASS_MODEL`.
