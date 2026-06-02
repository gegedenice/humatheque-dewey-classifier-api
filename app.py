"""FastAPI service for zero-shot Dewey classification with GLiClass."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.concurrency import run_in_threadpool
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

load_dotenv()


GLICLASS_MODEL = os.getenv("GLICLASS_MODEL", "knowledgator/gliclass-modern-large-v3.0")
GLICLASS_DEVICE = os.getenv("GLICLASS_DEVICE", "cpu")
GLICLASS_DTYPE = os.getenv("GLICLASS_DTYPE", "float32")
DEFAULT_CLASSIFICATION_TYPE = os.getenv("GLICLASS_CLASSIFICATION_TYPE", "multi-label")
DEFAULT_THRESHOLD = float(os.getenv("GLICLASS_THRESHOLD", "0.5"))
DEFAULT_MAX_LENGTH = int(os.getenv("GLICLASS_MAX_LENGTH", "1024"))
API_KEY = os.getenv("CLASSIFICATION_API_KEY", os.getenv("API_KEY", ""))

# Default labels are the Dewey divisions (main classes and their hundred-level
# subdivisions). Clients normally post their own `labels` list (discipline label
# + Dewey code); this default only keeps the endpoint usable without an explicit
# taxonomy.
DEFAULT_LABELS: list[dict[str, str]] = [
    {"000": "Informatique, information, généralités"},
    {"004": "Informatique"},
    {"020": "Bibliothéconomie et sciences de l'information"},
    {"060": "Organisations générales et muséologie"},
    {"070": "Médias d'information, journalisme, édition"},
    {"090": "Manuscrits et livres rares"},
    {"100": "Philosophie, psychologie"},
    {"110": "Métaphysique"},
    {"120": "Epistémologie, causalité, genre humain"},
    {"130": "Phénomènes paranormaux, pseudosciences"},
    {"140": "Les divers systèmes et écoles philosophiques"},
    {"150": "Psychologie"},
    {"160": "Logique"},
    {"170": "Morale (éthique)"},
    {"180": "Philosophie de l'Antiquité, du Moyen Âge, de l'Orient"},
    {"190": "Philosophie occidentale moderne et philosophies non orientales"},
    {"200": "Religion"},
    {"210": "Philosophie et théorie de la religion"},
    {"220": "Bible"},
    {"230": "Théologie chrétienne"},
    {"240": "Théologie morale et pratiques chrétiennes"},
    {"250": "Eglises locales, ordres religieux chrétiens"},
    {"260": "Théologie chrétienne et société, ecclésiologie"},
    {"270": "Histoire et géographie du christianisme et de l'Eglise chrétienne"},
    {"280": "Confessions et sectes de l'Eglise chrétienne"},
    {"290": "Autres religions"},
    {"300": "Sciences sociales, sociologie, anthropologie"},
    {"310": "Statistiques générales"},
    {"320": "Science politique"},
    {"330": "Economie"},
    {"340": "Droit"},
    {"350": "Administration publique. Arts et science militaires"},
    {"360": "Problèmes et services sociaux"},
    {"370": "Education et enseignement"},
    {"380": "Commerce, communications, transports"},
    {"390": "Ethnologie"},
    {"400": "Langues et linguistique"},
    {"410": "Linguistique générale"},
    {"420": "Langue anglaise. Anglo-saxon"},
    {"430": "Langues germaniques. Allemand"},
    {"440": "Langues romanes. Français"},
    {"450": "Langues italienne, roumaine, rhéto-romane"},
    {"460": "Langues espagnole et portugaise"},
    {"470": "Langues italiques. Latin"},
    {"480": "Langues helléniques. Grec classique"},
    {"490": "Autres langues"},
    {"500": "Sciences de la nature et mathématiques"},
    {"510": "Mathématiques"},
    {"520": "Astronomie, cartographie, géodésie"},
    {"530": "Physique"},
    {"540": "Chimie, minéralogie, cristallographie"},
    {"550": "Sciences de la terre"},
    {"560": "Paléontologie. Paléozoologie"},
    {"570": "Sciences de la vie, biologie, biochimie"},
    {"580": "Plantes. Botanique"},
    {"590": "Animaux. Zoologie"},
    {"600": "Technologie (Sciences appliquées)"},
    {"610": "Médecine et santé"},
    {"620": "Sciences de l'ingénieur"},
    {"630": "Agronomie, agriculture et médecine vétérinaire"},
    {"640": "Economie domestique. Vie familiale"},
    {"650": "Gestion et organisation de l'entreprise"},
    {"660": "Génie chimique, technologies alimentaires"},
    {"670": "Fabrication industrielle"},
    {"680": "Fabrication de produits à usages spécifiques"},
    {"690": "Bâtiments"},
    {"700": "Arts. Beaux-arts et arts décoratifs"},
    {"710": "Urbanisme"},
    {"720": "Architecture"},
    {"730": "Arts plastiques. Sculpture"},
    {"740": "Dessin. Arts décoratifs"},
    {"750": "Peinture"},
    {"760": "Arts graphiques"},
    {"770": "Photographie et les photographies, art numérique"},
    {"780": "Musique"},
    {"790": "Arts du spectacle, loisirs"},
    {"796": "Sport"},
    {"800": "Histoire et critique littéraires, rhétorique"},
    {"810": "Littérature américaine en anglais"},
    {"820": "Littératures anglaise et anglo-saxonne"},
    {"830": "Littérature allemande"},
    {"840": "Littérature de langues romanes. Littérature française"},
    {"850": "Littérature italienne"},
    {"860": "Littératures espagnole et portugaise"},
    {"870": "Littérature latine"},
    {"880": "Littérature grecque"},
    {"890": "Littératures des autres langues"},
    {"900": "Géographie et histoire"},
    {"910": "Géographie et voyages"},
    {"920": "Biographies générales, généalogie, emblèmes"},
    {"930": "Histoire ancienne et préhistoire"},
    {"940": "Histoire moderne et contemporaine de l'Europe"},
    {"944": "Histoire générale de la France"},
    {"950": "Histoire générale de l'Asie, Orient, Extrême-Orient"},
    {"960": "Histoire générale de l'Afrique"},
    {"970": "Histoire générale de l'Amérique du Nord"},
    {"980": "Histoire générale de l'Amérique du Sud"},
    {"990": "Histoire générale des autres parties du monde, des mondes extraterrestres. Iles du Pacifique"},
]


app = FastAPI(
    title="Humatheque Classification API",
    version="0.1.0",
    description=(
        "Zero-shot classification of academic text against a Dewey taxonomy using the "
        "GLiClass `gliclass-modern-large-v3.0` model. Clients post the text and a list "
        "of `{dewey_code: discipline_label}` entries; only the discipline labels are "
        "sent to the model, and the response re-attaches the matching Dewey code to "
        "each scored class."
    ),
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(api_key: str | None = Security(api_key_header)) -> None:
    if API_KEY and api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


class LabelScore(BaseModel):
    dewey: str | None = Field(None, description="Dewey code mapped back from the label.")
    label: str = Field(..., description="Discipline label scored by the model.")
    score: float = Field(..., description="Model confidence for the label.")


class TextClassification(BaseModel):
    text: str
    classes: list[LabelScore]


class ClassifyRequest(BaseModel):
    text: str | list[str] = Field(
        ...,
        description="Text to classify, or a list of texts for batch classification.",
    )
    labels: list[dict[str, str]] = Field(
        default_factory=lambda: list(DEFAULT_LABELS),
        description=(
            "Candidate labels as a list of single-entry dicts mapping a Dewey code to "
            "its discipline label, for example [{'004': 'Informatique'}]. Only the "
            "label text is sent to the model; the Dewey code is re-attached in the "
            "response."
        ),
    )
    threshold: float = Field(DEFAULT_THRESHOLD, ge=0.0, le=1.0)
    classification_type: str = Field(
        DEFAULT_CLASSIFICATION_TYPE,
        description="`multi-label` or `single-label`.",
    )
    top_k: int | None = Field(
        None,
        ge=1,
        description="Optional cap on the number of returned classes per text.",
    )


class ClassifyResponse(BaseModel):
    source: str
    model: str
    classification_type: str
    threshold: float
    count: int
    results: list[TextClassification]


def torch_dtype(name: str) -> Any:
    import torch

    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in mapping:
        raise HTTPException(status_code=500, detail=f"Unsupported GLICLASS_DTYPE {name!r}.")
    return mapping[name]


@lru_cache(maxsize=4)
def get_pipeline(classification_type: str) -> Any:
    """Build and cache a GLiClass pipeline per classification type.

    The model and tokenizer are heavy to load, so each (model, classification_type)
    pipeline is memoized and reused across requests.
    """
    from gliclass import GLiClassModel, ZeroShotClassificationPipeline
    from transformers import AutoTokenizer

    model = GLiClassModel.from_pretrained(GLICLASS_MODEL, dtype=torch_dtype(GLICLASS_DTYPE))
    tokenizer = AutoTokenizer.from_pretrained(GLICLASS_MODEL)
    return ZeroShotClassificationPipeline(
        model,
        tokenizer,
        classification_type=classification_type,
        device=GLICLASS_DEVICE,
        max_length=DEFAULT_MAX_LENGTH,
    )


def parse_labels(labels: list[dict[str, str]]) -> tuple[list[str], dict[str, list[str]]]:
    """Split the `{code: label}` entries into model labels and a label->codes map.

    Only the discipline label text is given to the model. The label->codes mapping
    lets the response re-attach the Dewey code(s) to each scored label. Duplicate
    label texts (same discipline under several codes) keep every code.
    """
    ordered_labels: list[str] = []
    label_to_codes: dict[str, list[str]] = {}
    for entry in labels:
        for code, label in entry.items():
            label = str(label).strip()
            code = str(code).strip()
            if not label:
                continue
            if label not in label_to_codes:
                label_to_codes[label] = []
                ordered_labels.append(label)
            if code and code not in label_to_codes[label]:
                label_to_codes[label].append(code)
    if not ordered_labels:
        raise HTTPException(status_code=400, detail="No usable labels provided.")
    return ordered_labels, label_to_codes


def to_label_scores(
    raw_results: list[dict[str, Any]],
    label_to_codes: dict[str, list[str]],
    top_k: int | None,
) -> list[LabelScore]:
    scored: list[LabelScore] = []
    for result in raw_results:
        label = str(result.get("label", ""))
        score = float(result.get("score", 0.0))
        codes = label_to_codes.get(label) or [None]
        for code in codes:
            scored.append(LabelScore(dewey=code, label=label, score=round(score, 4)))
    scored.sort(key=lambda item: item.score, reverse=True)
    if top_k is not None:
        scored = scored[:top_k]
    return scored


def classify(payload: ClassifyRequest) -> dict[str, Any]:
    texts = [payload.text] if isinstance(payload.text, str) else list(payload.text)
    if not texts or all(not text.strip() for text in texts):
        raise HTTPException(status_code=400, detail="Provide non-empty text.")

    ordered_labels, label_to_codes = parse_labels(payload.labels)

    pipeline = get_pipeline(payload.classification_type)
    raw_batches = pipeline(texts, ordered_labels, threshold=payload.threshold)

    results = [
        TextClassification(
            text=text,
            classes=to_label_scores(raw, label_to_codes, payload.top_k),
        )
        for text, raw in zip(texts, raw_batches)
    ]

    return {
        "source": "gliclass_classification",
        "model": GLICLASS_MODEL,
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

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
