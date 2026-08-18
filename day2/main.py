import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("API KEY NOT FOUND")

client = Groq(api_key = api_key)

model = "openai/gpt-oss-120b"

prompt = "Suggest a name for my food app"

message_system = {
    "role": "system",
    "content": "You are a brand manager who suggest names for my food app. The name should be in one word"
}

message = {
    "role" : "user",
    "content": prompt
}

messages = [message_system, message]

response = client.chat.completions.create(model=model, messages=messages, temperature=1.75)
print(response.choices[0].message.content)