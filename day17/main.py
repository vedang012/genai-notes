"""A small Groq agent with Tavily web search and a safe math calculator."""

import ast
import json
import operator
import os
from typing import Any

from dotenv import load_dotenv
from groq import Groq
from tavily import TavilyClient


load_dotenv()

groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


def web_search(query: str) -> str:
    """Search the web with Tavily and return concise source-backed context."""
    if not query.strip():
        return "Please provide a non-empty search query."

    response = tavily.search(
        query=query[:400],
        search_depth="basic",
        max_results=5,
        include_answer=True,
    )

    results = []
    for result in response.get("results", []):
        results.append(
            {
                "title": result.get("title"),
                "url": result.get("url"),
                "content": result.get("content", "")[:1200],
            }
        )

    return json.dumps(
        {
            "answer": response.get("answer"),
            "results": results,
        },
        ensure_ascii=False,
    )


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _evaluate_math(node: ast.AST) -> int | float:
    if isinstance(node, ast.Expression):
        return _evaluate_math(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate_math(node.left)
        right = _evaluate_math(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("Exponent is too large.")
        return _BINARY_OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_evaluate_math(node.operand))
    raise ValueError("Only numbers and basic arithmetic operators are allowed.")


def calculate(expression: str) -> str:
    """Calculate a basic arithmetic expression without executing arbitrary code."""
    if not expression.strip():
        return "Please provide a non-empty expression."
    try:
        tree = ast.parse(expression, mode="eval")
        result = _evaluate_math(tree)
        return str(result)
    except ZeroDivisionError:
        return "Error: division by zero."
    except (SyntaxError, ValueError, TypeError, OverflowError) as exc:
        return f"Error: {exc}"


# Groq uses this OpenAI-compatible tool schema format.
tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web with Tavily and return relevant current information and sources.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The web search query.",
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a basic arithmetic expression using numbers and +, -, *, /, //, %, **, and parentheses.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A basic arithmetic expression, for example '(12 + 8) / 4'.",
                    }
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
        },
    },
]

_TOOL_FUNCTIONS = {
    "web_search": web_search,
    "calculate": calculate,
}


def ask_agent(user_query: str) -> dict[str, Any]:
    """Send one query to Groq and return the answer plus a transparent tool trace.."""
    if not user_query.strip():
        return {
            "answer": "Please enter a question.",
            "thinking": "The input was empty, so no tool was selected.",
            "tools_used": [],
        }

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Use web_search for current or factual web information "
                "and calculate for arithmetic. Explain the result clearly and do not invent sources."
            ),
        },
        {"role": "user", "content": user_query},
    ]
    tools_used: list[dict[str, Any]] = []

    for _ in range(5):
        completion = groq.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        assistant_message = completion.choices[0].message

        if not assistant_message.tool_calls:
            if tools_used:
                selected = ", ".join(item["name"] for item in tools_used)
                thinking = f"I selected and executed these tool(s): {selected}."
            else:
                thinking = "No tool was needed for this request."
            return {
                "answer": assistant_message.content or "I could not generate a response.",
                "thinking": thinking,
                "tools_used": tools_used,
            }

        messages.append(
            {
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in assistant_message.tool_calls
                ],
            }
        )

        for call in assistant_message.tool_calls:
            arguments: dict[str, Any] = {}
            try:
                arguments = json.loads(call.function.arguments or "{}")
                function = _TOOL_FUNCTIONS[call.function.name]
                result = function(**arguments)
            except (KeyError, TypeError, json.JSONDecodeError, ValueError) as exc:
                result = f"Tool error: {exc}"

            tools_used.append(
                {
                    "name": call.function.name,
                    "arguments": arguments,
                }
            )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": str(result),
                }
            )

    return {
        "answer": "The agent reached its tool-call limit before producing an answer.",
        "thinking": "The model requested tools repeatedly and exceeded the safety limit.",
        "tools_used": tools_used,
    }


def main() -> None:
    user_query = input("You: ")
    print(json.dumps(ask_agent(user_query), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
