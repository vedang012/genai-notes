import os
import json

from time import sleep
from dotenv import load_dotenv
from groq import Groq

# load api key from the env 

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("Where the hell is API key")

# groq client setup

client = Groq(api_key=api_key)
model = "openai/gpt-oss-120b"

# tools

def calculator_tool(expression):
    try:
        return eval(expression)
    except:
        return "calc error"

def get_product_price(product):

    products = {
        "iphone17": 50000,
        "iphone16": 40000
    }

    product = product.lower().replace(" ", "")

    return products.get(product, 0)

tools = {
    "get_product_price": get_product_price,
    "calculator_tool": calculator_tool
}

# system prompt

system_prompt = """
You are a shopping assistant.

You have access to these tools:

1. get_product_price(product)
2. calculator_tool(expression)

You MUST respond with VALID JSON ONLY.

NEVER output Markdown.
NEVER use ```json.
NEVER output text outside the JSON object.

You have exactly two possible response types.

--------------------------------------------------
WHEN YOU NEED TO USE A TOOL
--------------------------------------------------

Return:

{
    "type": "tool_call",
    "tool": "TOOL_NAME",
    "argument": "TOOL_ARGUMENT",
    "thinking": "A brief summary of what you need to do next."
}

Example:

{
    "type": "tool_call",
    "tool": "get_product_price",
    "argument": "iphone17",
    "thinking": "I need to find the price of the iPhone 17."
}

Example:

{
    "type": "tool_call",
    "tool": "calculator_tool",
    "argument": "5000 - 1000",
    "thinking": "I need to calculate how much money will remain after buying the iPhone 17."
}

--------------------------------------------------
WHEN THE TASK IS COMPLETE
--------------------------------------------------

Return:

{
    "type": "final_answer",
    "answer": "YOUR ANSWER",
    "thinking": "A brief summary of why the task is complete."
}

Example:

{
    "type": "final_answer",
    "answer": "The iPhone 17 costs ₹50,000 and you will have ₹0 left.",
    "thinking":"I have all the information needed to answer"
}

--------------------------------------------------
RULES
--------------------------------------------------

1. Return exactly ONE JSON object.
2. The JSON must always be valid.
3. You can call only ONE tool at a time.
4. Never invent or guess a tool result.
5. After calling a tool, wait for the Observation.
6. Use the Observation to decide what to do next.
7. When no more tools are needed, return a final_answer.
8. The "tool" field must be exactly one of:
   "get_product_price"
   "calculator_tool"
9. For get_product_price, pass the product name as the argument.
10. For calculator_tool, pass a mathematical expression as the argument.
11. Even if a product is not known, call get_product_price rather than guessing its price.


"""

# AI AGENT


def agent(question):
    messages = [

        {
            "role": "system",
            "content": system_prompt
        },

        {
            "role": "user",
            "content": question
        }

    ]


    for step in range(5):

        print("\n------------------")
        print("STEP", step + 1)
        print("------------------")

        response = client.chat.completions.create(
            model = model,
            messages = messages,
            temperature = 0,
            response_format={"type": "json_object"}
        )

        answer = response.choices[0].message.content

        # print("RAW RESPONSE:")
        # print(repr(answer))

        data = json.loads(answer)


        print(data["thinking"])


        if data["type"] == "final_answer":
            print(data["answer"])
            break

        elif data["type"] == "tool_call":
            tool_name = data["tool"]
            arguement = data["argument"]

            if tool_name in tools:
                observation = tools[tool_name](arguement)
            else:
                observation = "tool not found"

        print(f"Observation : {observation}")

        # Add llm response to the memory

        messages.append(
            {
                "role": "assistant",
                "content" : answer
            }
        )

        
        messages.append(
            {
                "role": "user",
                "content": f"Observation: {observation}"
            }
        )


        sleep(5)

prompt="""
I have 100000 rupees. What is the price of an iphone 16?
and how much money will I have left?
"""

agent(prompt)

