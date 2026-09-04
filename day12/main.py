import os
from groq import Groq
from dotenv import load_dotenv

import numpy as np
from sentence_transformers import SentenceTransformer

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("API KEY not found or is invalid")

client = Groq(api_key = api_key)

groq_model = "openai/gpt-oss-120b"
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# this is our knowledge base. we convert it into embeddings

documents = [
    "Employees receive 24 days of paid leave per year.",
   
    "Employees work from the office on Tuesday, Wednesday and Thursday. "
    "Monday and Friday are optional work-from-home days.",
   
    "Employees receive Rs 3000 per month for gym reimbursement.",
   
    "Employees can claim Rs 2000 per month for home internet.",
   
    "Employees have a 90 day notice period."
]

docuement_embeddings = embedding_model.encode(documents)


def cosine_similarity(a, b):
    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


def retrieve(query_embedding):
    scores = []
    for i, document in enumerate(docuement_embeddings):
        score = cosine_similarity(query_embedding, document)
        scores.append((score, documents[i]))
    scores.sort(reverse = True)
    return scores[0]

def ask_llm(query, context):

    messages = [
        {
            "role": "system",
            "content": f"answer based on this context : {context}"
        },

        {
            "role": "user",
            "content": query
        }
    ]

    response = client.chat.completions.create(model = groq_model, messages = messages)

    return response.choices[0].message.content


query = "How many vacations will be there?"
query_embedding = embedding_model.encode(query)

score, context = retrieve(query_embedding)
print(score, context)

response = ask_llm(query, context)
print("\n", response)


