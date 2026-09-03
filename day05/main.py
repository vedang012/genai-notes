import os
import json

from dotenv import load_dotenv
from groq import Groq
from models import Resume
import parser

# Configuration

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found")

MODEL = "openai/gpt-oss-120b"

client = Groq(api_key=api_key)

resume_model = Resume


# Input

resume_text = parser.extract_text("resume.pdf")


# Prompt

schema = resume_model.model_json_schema()

system_prompt = f"""
Extract the information from the resume.

Return JSON matching this schema:

{schema}
"""

user_prompt = f"""
Resume Text:

{resume_text}
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

resume = resume_model(**data)


# Output

print(resume.name)
print(resume.email)
print(resume.mobile)
print(resume.skills)
print(resume.education)
print(resume.experience)
print(resume.projects)
