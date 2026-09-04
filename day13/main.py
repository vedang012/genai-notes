import os

from dotenv import load_dotenv
from groq import Groq
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

COLLECTION_NAME = "mydata"

# Connect to Qdrant
qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to Groq
groq_client = Groq(
    api_key=GROQ_API_KEY
)


def search(query, top_k=5):

    query_vector = model.encode(query).tolist()

    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True
    ).points

    return results


def ask_groq(question, context):

    system_prompt = f"""
You are a travel guide for Maharashtra.

Answer the user's question using ONLY the provided context.

Rules:
- Do not use outside knowledge.
- Do not invent facts.
- If the answer cannot be found in the context, clearly say so.
- Keep the answer concise and useful.

Context:
{context}
"""

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": question
        }
    ]

    response = groq_client.chat.completions.create(
        messages=messages,
        model="openai/gpt-oss-120b"
    )

    return response.choices[0].message.content


# Query
query = "I'm in Pune and I wanna go somewhere uphill."

results = search(query)

# Extract retrieved context
context = "\n\n".join(
    result.payload["text"]
    for result in results
)

# Debug retrieval
for result in results:
    print("=" * 60)
    print("Score:", result.score)
    print(result.payload["text"])

print("\nANSWER:\n")

print(ask_groq(query, context))