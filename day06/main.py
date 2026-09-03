import os
from dotenv import load_dotenv
from groq import Groq

# Config

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ API KEY NOT FOUND")

MODEL = "openai/gpt-oss-120b"

client = Groq(api_key = api_key)

def llm_prompt(prompt):

    message = {
        "role": "user",
        "content": prompt
    }

    messages = [message]

    response = client.chat.completions.create(
        model = MODEL,
        messages = messages
    )

    print(response.choices[0].message.content)


user_complaint = "Operating system is not booting up"


prompt = f"""

# Role : You are a technical assistant at a laptop company.

# Task : You have to classify the user complaint.

# Constraint : Possible classifications are - 1. Billing issue, 2. Technical issue, 3. Product return/replacement

# Output Format : Write only the classified category as the output.. nothing else

# Example : 'Hinge is broken' is a technical issue

# Fallback : Write 'OTHER' for unrelated issues

User complaint : {user_complaint}

"""

llm_prompt(prompt)

