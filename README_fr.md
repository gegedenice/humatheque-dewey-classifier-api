# API de classification Humatheque

Service FastAPI **local** qui attribue une **classe Dewey** à un texte académique
court (titres, parfois résumés) par **similarité sémantique** — entièrement hors
ligne, avec des modèles à poids ouverts et sans API payante.

Le service est conçu pour une chaîne de catalogage où un texte doit recevoir une
affectation disciplinaire / Dewey. Un client envoie le texte ; l'API l'encode
avec un modèle local d'embeddings multilingue, le compare à la taxonomie Dewey et
renvoie les classes les plus proches avec un score de similarité. Chaque classe
renvoyée porte `dewey` + `label` + `score`.

## Fonctionnement

Les classes forment une taxonomie Dewey fixe et **faisant autorité**, stockée
dans `taxonomy.json`. Chaque entrée possède trois champs :

```json
{"code": "980", "label": "Histoire générale de l'Amérique du Sud",
 "description": "Histoire de l'Amérique du Sud et latine, Argentine, Brésil, Buenos Aires, indépendances sud-américaines"}
```

- `code` + `label` **font autorité** — c'est exactement ce que l'API renvoie.
- `description` est **interne** : un ensemble enrichi de mots-clés (noms de lieux,
  époques, domaines, synonymes) utilisé uniquement pour construire l'embedding de
  la classe.

Pourquoi l'enrichissement est essentiel : les titres entrants sont brefs et très
spécifiques (par ex. un sujet de thèse) et doivent être rattachés à une catégorie
*plus large*. Un simple libellé ne peut pas relier « Buenos Aires, 1829 » à
« Amérique du Sud » — c'est le vocabulaire discriminant de `description` qui rend
cette correspondance possible. **On améliore la précision en éditant
`taxonomy.json`, pas en réglant le modèle**, et un·e bibliothécaire peut le faire
sans toucher au Python.

### Logique de classification

`POST /classify` :

1. Encoder chaque classe Dewey une fois au démarrage sous la forme
   `"{label}. {description}"` (avec le préfixe « passage » du modèle). L'index est
   construit une fois par processus puis mis en cache.
2. Encoder le texte entrant (avec le préfixe « query »).
3. Classer les classes par similarité cosinus, écarter ce qui est sous le
   `threshold`, trier et renvoyer les `top_k` premières.

`multi-label` (par défaut) renvoie jusqu'à `top_k` classes ; `single-label`
renvoie la meilleure classe.

### Méthodes de classification

La stratégie se choisit par requête via le champ `method` (défaut
`CLASSIFICATION_METHOD`) :

- **`local`** (défaut) — le bi-encodeur local décrit ci-dessus. Entièrement hors
  ligne, classe toute la taxonomie par similarité cosinus. Aucun appel externe.
- **`albert`** — une chaîne en deux temps *récupération puis reclassement*,
  servie par l'[API Albert](https://albert.api.etalab.gouv.fr) (la passerelle LLM
  souveraine de l'État français). Un **bi-encodeur** distant (`BAAI/bge-m3`)
  réduit la taxonomie à un vivier de `RERANK_CANDIDATES` candidats, puis un
  **cross-encodeur** distant (`BAAI/bge-reranker-v2-m3`) re-note la description
  enrichie de chaque candidat face au texte. Les cross-encodeurs lisent la requête
  et le document *ensemble*, le reclassement est donc en général plus net que le
  seul cosinus du bi-encodeur. Nécessite `ALBERT_API_KEY`.

Les deux méthodes partagent le même `taxonomy.json` et les mêmes exemples k-NN :
enrichir les descriptions améliore les deux. Attention, `albert` renvoie des
**scores de pertinence du reranker**, et non des similarités cosinus ; les valeurs
ne sont donc pas comparables d'une méthode à l'autre.

### Modèle

Le modèle par défaut est
[`intfloat/multilingual-e5-large`](https://huggingface.co/intfloat/multilingual-e5-large),
servi via `sentence-transformers`. Il est multilingue (performant en français),
fonctionne sur CPU et reste entièrement local. On peut le remplacer via
`EMBEDDING_MODEL` (par ex. `BAAI/bge-m3`, ou `intfloat/multilingual-e5-base` pour
une latence plus faible).

Les modèles e5/bge utilisent des préfixes asymétriques — le texte recherché est
une `query: ` et les descriptions de classes sont des `passage: `. Ces préfixes
sont configurables et **doivent conserver leur espace final** (d'où les
guillemets dans `.example.env`).

### Amélioration dans le temps (k-NN)

L'index peut aussi se comparer à des exemples déjà catalogués. Indiquez dans
`EXAMPLES_PATH` un fichier JSON d'affectations confirmées :

```json
[
  {"text": "Étude des algorithmes d'apprentissage automatique", "code": "004"},
  {"text": "Histoire politique de Buenos Aires au XIXe siècle", "code": "980"}
]
```

Chaque exemple est encodé par classe et combiné à la similarité de la description
ainsi :
`final = max(similarité_description, EMBEDDING_EXAMPLE_WEIGHT × meilleure_similarité_exemple)`,
de sorte que les affectations confirmées ne peuvent qu'améliorer les résultats.
Laissez `EXAMPLES_PATH` vide pour désactiver cette fonction.

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
  "text": "Handbook on large language models and embeddings models.",
  "codes": null,
  "threshold": 0.0,
  "classification_type": "multi-label",
  "top_k": 5,
  "method": "local"
}
```

| Champ | Requis | Description |
|---|---:|---|
| `text` | oui | Une chaîne, ou une liste de chaînes pour un traitement par lot |
| `codes` | non | Sous-ensemble optionnel de codes Dewey pour restreindre les candidats (ex. `["004","510"]`) ; par défaut toute la taxonomie. Les codes inconnus sont ignorés |
| `threshold` | non | Score minimal pour renvoyer une classe, défaut `0.0` |
| `classification_type` | non | `multi-label` (défaut, jusqu'à `top_k`) ou `single-label` (meilleure classe) |
| `top_k` | non | Plafond de classes renvoyées par texte ; défaut depuis `CLASSIFICATION_TOP_K` (`5`) |
| `method` | non | `local` (défaut, bi-encodeur local) ou `albert` (API Albert : récupération bge-m3 + reclassement bge-reranker-v2-m3) ; défaut depuis `CLASSIFICATION_METHOD` |

Exemple :

```bash
curl -s localhost:8000/classify -H 'Content-Type: application/json' -d '{
  "text": "Handbook on large language models and embeddings models.",
  "top_k": 3
}'
```

Forme de la réponse :

```jsonc
{
  "source": "embedding_classification",
  "method": "local",
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

La méthode `albert` renvoie **la même structure** ; seules les métadonnées et la
signification du `score` changent. Le `score` est le `relevance_score` du
cross-encodeur, repris tel quel — ce sont des scores de pertinence du reranker, et
non des similarités cosinus : ils ne sont donc pas comparables aux scores `local` :

```jsonc
{
  "source": "albert_rerank_classification",
  "method": "albert",
  "model": "BAAI/bge-m3 + BAAI/bge-reranker-v2-m3",
  "classification_type": "multi-label",
  "threshold": 0.0,
  "count": 1,
  "results": [
    {
      "text": "Handbook on large language models and embeddings models.",
      "classes": [
        {"dewey": "004", "label": "Informatique", "score": 0.97},
        {"dewey": "410", "label": "Linguistique générale", "score": 0.12},
        {"dewey": "510", "label": "Mathématiques", "score": 0.04}
      ]
    }
  ]
}
```

Le traitement par lot envoie une liste de textes et renvoie une entrée par texte :

```json
{"text": ["Premier texte à classer.", "Second texte à classer."], "top_k": 3}
```

> **À propos des scores :** ce sont des similarités cosinus, pas des probabilités
> calibrées. Avec les modèles de type e5, elles sont élevées et resserrées
> (≈0,7–0,9) même pour des correspondances faibles ; il faut donc les lire comme
> un *classement*. `threshold` vaut `0.0` par défaut ; privilégiez `top_k` et une
> validation humaine plutôt qu'un seuil strict.

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
| `CLASSIFICATION_API_KEY` | vide | Clé API optionnelle (`API_KEY` est aussi acceptée) |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-large` | Identifiant du modèle d'embeddings |
| `EMBEDDING_DEVICE` | `cpu` | Périphérique d'inférence (`cpu`, `cuda:0`, ...) |
| `HF_TOKEN` | vide | Jeton Hugging Face optionnel pour télécharger le modèle (`HUGGING_FACE_HUB_TOKEN` est aussi accepté) ; requis pour les modèles privés/restreints |
| `EMBEDDING_QUERY_PREFIX` | `"query: "` | Préfixe du texte recherché (propre au modèle ; conserver l'espace final) |
| `EMBEDDING_PASSAGE_PREFIX` | `"passage: "` | Préfixe des descriptions de classes / exemples (conserver l'espace final) |
| `TAXONOMY_PATH` | `taxonomy.json` | Chemin de la taxonomie Dewey faisant autorité |
| `EXAMPLES_PATH` | vide | JSON optionnel d'exemples catalogués `{text, code}` pour le k-NN |
| `EMBEDDING_EXAMPLE_WEIGHT` | `1.0` | Poids de la meilleure similarité d'exemple lors de la combinaison |
| `CLASSIFICATION_TOP_K` | `5` | Plafond par défaut de classes renvoyées par texte |
| `CLASSIFICATION_THRESHOLD` | `0.0` | Score minimal par défaut |
| `CLASSIFICATION_TYPE` | `multi-label` | Type de classification par défaut |
| `CLASSIFICATION_METHOD` | `local` | Stratégie par défaut (`local` ou `albert`) |
| `ALBERT_API_KEY` | vide | Jeton bearer de l'API Albert ; requis pour `method=albert` |
| `ALBERT_BASE_URL` | `https://albert.api.etalab.gouv.fr/v1` | URL de base de l'API Albert |
| `ALBERT_EMBEDDING_MODEL` | `BAAI/bge-m3` | Bi-encodeur distant Albert (étape de récupération) |
| `ALBERT_RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | Cross-encodeur distant Albert (étape de reclassement) |
| `ALBERT_QUERY_PREFIX` | vide | Préfixe « query » optionnel pour le bi-encodeur Albert (bge-m3 n'en a pas besoin) |
| `ALBERT_PASSAGE_PREFIX` | vide | Préfixe « passage » optionnel pour le bi-encodeur Albert |
| `ALBERT_TIMEOUT` | `30` | Délai d'attente (secondes) par appel à l'API Albert |
| `RERANK_CANDIDATES` | `20` | Taille du vivier de candidats du bi-encodeur transmis au reranker (`albert`) |

## Exécution locale

```bash
cp .example.env .env
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

La première requête télécharge le modèle d'embeddings depuis Hugging Face et
construit l'index de la taxonomie ; les requêtes suivantes réutilisent le modèle
chargé.

## Docker

```bash
docker build -t humatheque-classification-api .
docker run --env-file .env -p 8000:8000 humatheque-classification-api
```

Le cache Hugging Face se trouve sous `/app/.cache/huggingface` ; montez un volume
à cet emplacement pour éviter de retélécharger le modèle à chaque démarrage du
conteneur.

## Notes d'exploitation

- Le modèle et l'index de la taxonomie sont chargés une fois et mis en cache pour
  la durée de vie du processus. **Redémarrez le service** pour prendre en compte
  les modifications de `taxonomy.json`, de `EXAMPLES_PATH` ou du modèle.
- L'encodage est limité par le CPU ; il s'exécute dans un threadpool pour ne pas
  bloquer la boucle d'événements, mais un worker unique traite une requête à la
  fois.
- Pour le débit, préférez les requêtes par lot (une liste de textes).
- La précision dépend des mots-clés `description` de `taxonomy.json` et des
  exemples catalogués optionnels — pas des paramètres du modèle.
