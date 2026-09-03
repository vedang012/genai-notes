from groq import Groq
from dotenv import load_dotenv

import os

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("API KEY not found or is invalid")

client = Groq(api_key = api_key)

model = "openai/gpt-oss-120b"

knowledge_base={
    "age" : " The age of pratyush is 25 years",
    "net worth" : "The net worth of pratyush is 2000"
}

# step 2 retreieval
def retrieve_info(question):
    question=question.lower()
    if "age" in question:
        return knowledge_base["age"]
    elif "net worth" in question:
        return knowledge_base["net worth"]
    else:
        return None


def ask_llm(question):

    context = retrieve_info(question)

    messages = [
        {
            "role": "system",
            "content": f"answer only based on this context : {context}"
        },

        {
            "role": "user",
            "content": question
        }
    ]

    response = client.chat.completions.create(model = model, messages = messages)

    return response.choices[0].message.content

question = "what is pratyush's age"

print(ask_llm(question))