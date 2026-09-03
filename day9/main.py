from groq import Groq
from dotenv import load_dotenv

import os

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("API KEY not found or is invalid")

client = Groq(api_key = api_key)

model = "openai/gpt-oss-120b"

prompt = "Explain how internet works"

messages = [
    {
        "role": "user",
        "content": prompt
    }
]

stream = client.chat.completions.create(messages = messages, model = model, stream=True)

for chunk in stream:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end = "", flush = True, )