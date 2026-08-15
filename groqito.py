import json
import os
from typing import Optional

from groq import Groq
from pydantic import BaseModel


class Translation(BaseModel):
    text: str
    comments: Optional[str] = None


def _client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Export it or put it in a local .env file."
        )
    return Groq(api_key=api_key)


def groq_translate(query, to_language):
    print(f"Translating '{query}' to {to_language}")

    chat_completion = _client().chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": (
                    f"You are a helpful assistant that translates a dirty OCR output text to a clean: {to_language} translation."
                    f"You will only reply with the translation text and nothing else in JSON."
                    f" The JSON object must use the schema: {json.dumps(Translation.model_json_schema(), indent=2)}"
                ),
            },
            {
                "role": "user",
                "content": f"Translate '{query}' from to {to_language}.",
            },
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        max_tokens=1024,
        stream=False,
        response_format={"type": "json_object"},
    )
    print(chat_completion.choices[0].message.content)
    return Translation.model_validate_json(chat_completion.choices[0].message.content)
