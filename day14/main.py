import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, MatchAny, PayloadSchemaType
from sentence_transformers import SentenceTransformer
from groq import Groq
import json

# load env

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# create groq client

groq_client = Groq(api_key = GROQ_API_KEY)


# connect to qdrant vectordb


qdrant_client = QdrantClient(
    url = QDRANT_URL,
    api_key = QDRANT_API_KEY
)

print("connected to the qdrant client")


# create collection

COLLECTION_NAME = "knowledge_filter"
EMBEDDING_SIZE = 384

# if the collection already exists delete it
if qdrant_client.collection_exists(collection_name = COLLECTION_NAME):
    print(f"existing collection {COLLECTION_NAME} found. deleting it")
    qdrant_client.delete_collection(collection_name = COLLECTION_NAME)

# create new

qdrant_client.create_collection(
    collection_name = COLLECTION_NAME,
    vectors_config = VectorParams(
        size = EMBEDDING_SIZE,
        distance = Distance.COSINE
    )
)

print(f"Created collection: {COLLECTION_NAME}")
print(f"Vector size: {EMBEDDING_SIZE}")
print("Distance: COSINE")

# Creates index for a given payload field. Indexed fields allow to perform filtered search operations faster.

qdrant_client.create_payload_index(
    collection_name = COLLECTION_NAME,
    field_name = "category",
    field_schema = PayloadSchemaType.KEYWORD
)

# load knowledge

with open("knowledge.json", "r", encoding = "utf-8") as file:
    documents = json.load(file)


# create embeddings 

print("loading embedding model")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("embedding model is now ready")

texts = []
for document in documents:
    texts.append(document["text"])

embeddings = model.encode(texts)

# create qdrant points

points = []

for i, embedding in enumerate(embeddings):

    point = PointStruct(
        id = i,
        vector = embedding,
        payload = documents[i]
    )

    points.append(point)

# upload to qdrant

qdrant_client.upsert(
    collection_name = COLLECTION_NAME,
    points = points,

)

print(f"uploaded {len(points)} documents to qdrant")

# search qdrant

def search(query, top_k):
    query_vector = model.encode(query).tolist()

    # search qdrant for similar vectors

    results = qdrant_client.query_points(
        collection_name = COLLECTION_NAME,
        query = query_vector,
        limit = top_k,
        with_payload = True
    ).points

    return results

def search_with_filter(query, top_k, filter):

    query_vector = model.encode(query).tolist()

    # search

    results = qdrant_client.query_points(
        collection_name = COLLECTION_NAME,
        query = query_vector,
        limit = top_k,
        with_payload = True,
        query_filter = filter
    ).points

    return results

reimbursement_filter = Filter(
    must = [
        FieldCondition(
            key = "category",
            match = MatchValue(value = "reimbursement")  
        )
    ]
)

# ask llm

def ask_llm(query, context):
    sys_prompt = f"""

you are a company HR assistant and your role is to solve employee's query with the help of given context. answer only based on the following context:

context = {context}

"""

    messages = [

        {
            "role": "system",
            "content": sys_prompt
        },

        {
            "role": "user",
            "content": query
        }

    ]

    results = groq_client.chat.completions.create(
        messages = messages,
        model = "openai/gpt-oss-120b",
        temperature = 0.15
    )

    response = results.choices[0].message.content

    return response

# let's test this shi

query = "how many holidays do i get"

results = search_with_filter(query = query, top_k = 3, filter = reimbursement_filter)

context_parts = []

for result in results:
    print(result)
    text = result.payload["text"]
    context_parts.append(text)

context = "\n\n".join(context_parts)

print(ask_llm(query, context))
