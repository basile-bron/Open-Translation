#todo pythondotenv import
import os

os.environ['GROQ_API_KEY'] = "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"

# groq_translation.py
import json
from typing import Optional

from groq import Groq
from pydantic import BaseModel

# Set up the Groq client

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# Model for the translation
class Translation(BaseModel):
    text: str
    comments: Optional[str] = None


# Translate text using the Groq API
def groq_translate(query, to_language):
    # Print the query and target language
    print(f"Translating '{query}' to {to_language}")
    
    # Create a chat completion
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": f"You are a helpful assistant that translates a dirty OCR output text to a clean: {to_language} translation."
                           f"You will only reply with the translation text and nothing else in JSON."
                           f" The JSON object must use the schema: {json.dumps(Translation.model_json_schema(), indent=2)}",
            },
            {
                "role": "user",
                "content": f"Translate '{query}' from to {to_language}."
            }
        ],
        model="mixtral-8x7b-32768",
        temperature=0.2,
        max_tokens=1024,
        stream=False,
        response_format={"type": "json_object"},
    )
    # Print the translation
    print(chat_completion.choices[0].message.content)
    # Return the translated text
    return Translation.model_validate_json(chat_completion.choices[0].message.content)


