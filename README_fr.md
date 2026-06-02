# API de classification Humatheque

Service FastAPI de classification *zero-shot* de texte académique selon une
taxonomie Dewey, à l'aide du modèle GLiClass `gliclass-modern-large-v3.0`.

Le service est conçu pour une chaîne de catalogage où un texte (un titre, un
résumé, ou du contenu extrait d'un document) doit recevoir une affectation de
discipline / Dewey. Un client envoie le texte et une taxonomie candidate, et
l'API renvoie les classes correspondantes avec leurs scores de confiance.

La taxonomie est fournie sous forme de liste d'entrées
`{code_dewey: libellé_discipline}`. Seul le libellé de discipline est envoyé au
modèle (GLiClass classe sur le texte du libellé), et la réponse réassocie le code
Dewey à chaque classe scorée : chaque classe renvoyée porte donc `dewey` +
`label` + `score`.

## Modèle

Le service utilise la librairie GLiClass et le modèle
[`knowledgator/gliclass-modern-large-v3.0`](https://huggingface.co/knowledgator/gliclass-modern-large-v3.0),
servi via `ZeroShotClassificationPipeline` :

```python
from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from transformers import AutoTokenizer
import torch

model = GLiClassModel.from_pretrained("knowledgator/gliclass-modern-large-v3.0", dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained("knowledgator/gliclass-modern-large-v3.0")
pipeline = ZeroShotClassificationPipeline(model, tokenizer, classification_type="multi-label", device="cpu")
```

Le pipeline est construit une seule fois par `classification_type` puis mis en
cache pour la durée de vie du processus. La première requête supporte le coût de
chargement du modèle ; les requêtes suivantes réutilisent le modèle chargé.
L'inférence s'exécute dans un *threadpool* pour ne pas bloquer la boucle
d'événements.

GLiClass gère nativement la classification d'un texte unique comme d'un lot, donc
ce service accepte soit une chaîne `text` unique, soit une liste de textes dans
une même requête.

## Logique de classification

`POST /classify` :

1. Normalise `text` en liste (chaîne unique ou lot).
2. Analyse `labels` (`[{code: libellé}, ...]`) en :
   - la liste ordonnée des libellés de discipline transmis au modèle ;
   - une table `libellé -> [codes Dewey]` servant à réassocier les codes dans la
     réponse.
3. Exécute le pipeline GLiClass avec le `threshold` demandé.
4. Pour chaque texte, renvoie les classes au-dessus du seuil sous la forme
   `dewey` + `label` + `score`, triées par score décroissant (limitées par
   `top_k` si fourni).

`multi-label` (par défaut) renvoie tous les libellés dont le score passe le
seuil. `single-label` renvoie la meilleure classe unique.

## Points d'accès

La documentation interactive est disponible sur `/docs`.

### `GET /health`

Vérification d'état du conteneur.

```json
{"ok": true}
```

### `POST /classify`

Corps de la requête :

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

| Champ | Requis | Description |
|---|---:|---|
| `text` | oui | Une chaîne, ou une liste de chaînes pour la classification par lot |
| `labels` | non | Liste d'entrées `{code_dewey: libellé_discipline}` ; par défaut, les divisions Dewey (classes principales et subdivisions de niveau cent) |
| `threshold` | non | Score minimal pour renvoyer une classe, défaut `0.5` |
| `classification_type` | non | `multi-label` (défaut) ou `single-label` |
| `top_k` | non | Limite optionnelle du nombre de classes renvoyées par texte |

`labels` n'envoie que le libellé de discipline au modèle ; le code Dewey est
réassocié à chaque résultat. Si plusieurs codes partagent le même libellé, tous
les codes sont renvoyés pour ce libellé.

Exemple :

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

Forme de la réponse :

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

La classification par lot envoie une liste de textes et renvoie une entrée par
texte dans `results` :

```json
{
  "text": ["Premier texte à classer.", "Second texte à classer."],
  "labels": [{"004": "Informatique"}, {"200": "Religion"}]
}
```

## Authentification

L'authentification est optionnelle. Définissez `CLASSIFICATION_API_KEY` (ou
`API_KEY`) ; les clients doivent alors envoyer :

```text
X-API-Key: <clé>
```

Si la clé est vide, le point d'accès est public.

## Variables d'environnement

Copiez `.example.env` vers `.env` et ajustez les valeurs.

| Variable | Défaut | Description |
|---|---|---|
| `PORT` | `8000` | Port du serveur HTTP |
| `CLASSIFICATION_API_KEY` | vide | Clé API optionnelle (`API_KEY` est aussi pris en compte) |
| `GLICLASS_MODEL` | `knowledgator/gliclass-modern-large-v3.0` | Identifiant du modèle GLiClass |
| `GLICLASS_DEVICE` | `cpu` | Périphérique d'inférence (`cpu`, `cuda:0`, ...) |
| `GLICLASS_DTYPE` | `float32` | Type du modèle (`float32`, `float16`, `bfloat16`) |
| `GLICLASS_CLASSIFICATION_TYPE` | `multi-label` | Type de classification par défaut |
| `GLICLASS_THRESHOLD` | `0.5` | Seuil de score par défaut |
| `GLICLASS_MAX_LENGTH` | `1024` | Longueur maximale en tokens par texte |

## Exécution locale

```bash
cp .example.env .env
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

La première requête télécharge le modèle depuis Hugging Face et le charge en
mémoire ; elle est donc plus lente que les suivantes.

## Docker

```bash
docker build -t humatheque-classification-api .
docker run --env-file .env -p 8000:8000 humatheque-classification-api
```

L'image Docker suit le même style de déploiement que les autres services
Humatheque : image Python *slim*, `requirements.txt`, utilisateur non *root*, et
`uvicorn app:app`. Le cache Hugging Face est situé sous
`/app/.cache/huggingface` ; montez un volume à cet emplacement pour éviter de
re-télécharger le modèle à chaque démarrage du conteneur.

## Notes opérationnelles

- Le modèle est chargé paresseusement à la première requête et mis en cache par
  `classification_type` pour la durée de vie du processus.
- L'inférence est limitée par le CPU ; elle s'exécute dans un *threadpool* pour
  ne pas bloquer la boucle d'événements, mais un seul *worker* traite un lot à la
  fois.
- Pour le débit, préférez les requêtes par lot (une liste de textes) plutôt que
  de nombreuses requêtes à texte unique, car GLiClass les traite en une seule
  passe.
- `gliclass-modern-large-v3.0` est la variante *large* ; pour un service à plus
  faible latence, envisagez un modèle GLiClass plus petit via `GLICLASS_MODEL`.
