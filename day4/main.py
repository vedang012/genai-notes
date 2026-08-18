import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("API KEY NOT FOUND")

client = Groq(api_key = api_key)

model = "openai/gpt-oss-120b"

class Ticket(BaseModel):
    name:str
    email:str
    issue:str

schema = Ticket.model_json_schema()

response_format = {
    "type": "json_object"
}

system_prompt = f"""
Extract the personal information from the ticket based on this schema and give me a json output. {schema}
"""

message_system = {
    "role": "system",
    "content": system_prompt
}


role = "user"

text = "Hello, My name is Vedang and my laptop is not working properly i want a replacement.. this is my email - vedang@fds.in"

prompt = f"""
    This is a customer ticket, extract the customer personal info. {text}
"""

message = {
    "role" : role,
    "content": prompt
}

messages = [message_system, message]

response = client.chat.completions.create(model=model, messages=messages, response_format=response_format)
 
answer = response.choices[0].message.content;

print(answer)


import json
raw_json = answer
data_file = json.loads(raw_json)

ticket = Ticket(**data_file)

print(ticket.name)
print(ticket.email)
print(ticket.issue)