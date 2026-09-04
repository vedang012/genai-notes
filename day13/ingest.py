import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = "mydata"
EMBEDDING_SIZE = 384

# Connect to Qdrant
qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

# Create collection if it doesn't exist
if not qdrant_client.collection_exists(COLLECTION_NAME):
    print("Creating collection...")

    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=EMBEDDING_SIZE,
            distance=Distance.COSINE
        )
    )

# Load knowledge
with open("travel_guide.txt", "r", encoding="utf-8") as file:
    text = file.read()

documents = [
    paragraph.strip()
    for paragraph in text.split("\n\n")
    if paragraph.strip()
]

print(f"Loaded {len(documents)} documents.")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create embeddings
print("Creating embeddings...")
embeddings = model.encode(documents)

# Create Qdrant points
points = []

for i, embedding in enumerate(embeddings):
    points.append(
        PointStruct(
            id=i + 1,
            vector=embedding.tolist(),
            payload={
                "text": documents[i]
            }
        )
    )

# Upload
print(f"Uploading {len(points)} points...")

qdrant_client.upsert(
    collection_name=COLLECTION_NAME,
    points=points
)

print("Ingestion complete!")