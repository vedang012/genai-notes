import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("API KEY NOT FOUND")

client = Groq(api_key = api_key)

model = "llama-3.3-70b-versatile"
role = "user"
prompt = "what is today's date"

message = {
    "role" : role,
    "content": prompt
}

messages = [message]

response = client.chat.completions.create(model=model, messages=messages)
print(response.choices[0].message.content)