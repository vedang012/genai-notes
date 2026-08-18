import os
import json

from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel


# Configuration

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found")

MODEL = "openai/gpt-oss-120b"

client = Groq(api_key=api_key)


# Data model

class Ticket(BaseModel):
    name: str
    email: str
    issue: str


# Input

text = """
Hello, My name is Vedang and my laptop is not working properly.
I want a replacement. This is my email - vedang@fds.in
"""


# Prompt

schema = Ticket.model_json_schema()

system_prompt = f"""
Extract the following information from the customer ticket.

Return JSON matching this schema:

{schema}
"""

user_prompt = f"""
Customer ticket:

{text}
"""


# LLM request

messages = [
    {
        "role": "system",
        "content": system_prompt
    },
    {
        "role": "user",
        "content": user_prompt
    }
]

response = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    response_format={"type": "json_object"}
)


# Parse + validate

answer = response.choices[0].message.content

data = json.loads(answer)

ticket = Ticket(**data)


# Output

print(ticket.name)
print(ticket.email)
print(ticket.issue)
