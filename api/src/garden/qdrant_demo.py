import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

# --- Config ---
COLLECTION = "jardinage"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"  # MÊME modèle qu'avant
QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")

# --- 1. Connexions ---
model = SentenceTransformer(MODEL_NAME)
client = QdrantClient(url=QDRANT_URL)

# --- 2. Créer la collection (repart de zéro à chaque run) ---
if client.collection_exists(COLLECTION):
    client.delete_collection(COLLECTION)

client.create_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

# --- 3. Nos documents jardinage ---
documents = [
    "Le mildiou provoque des taches brunes sur les feuilles de tomate.",
    "Arroser les tomates au pied, jamais sur le feuillage, pour éviter les maladies.",
    "La taille des rosiers se fait en fin d'hiver, avant la reprise.",
    "Le compost mûr enrichit le sol et améliore sa structure.",
]

# --- 4. Encoder + insérer ---
points = [
    PointStruct(
        id=i,
        vector=model.encode(doc).tolist(),
        payload={"texte": doc},
    )
    for i, doc in enumerate(documents)
]
client.upsert(collection_name=COLLECTION, points=points)
print(f"{len(points)} documents indexés.\n")

# --- 5. Poser une question ---
question = "mes tomates ont des taches"
q_vector = model.encode(question).tolist()

resultats = client.search(
    collection_name=COLLECTION,
    query_vector=q_vector,
    limit=2,
)

# --- 6. Afficher ---
print(f"Question : {question}\n")
for r in resultats:
    print(f"[score {r.score:.3f}]  {r.payload['texte']}")