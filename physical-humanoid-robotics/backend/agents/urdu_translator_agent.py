from typing import Dict, Any
from core.config import settings
from openai import OpenAI

class UrduTranslatorAgent:
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)

    def translate_to_urdu(self, text: str, context: str = None) -> Dict[str, Any]:
        """
        Translate content to Urdu with context awareness.
        """
        prompt = f"""
        Translate the following text to Urdu:
        {text}

        Context: {context or 'No context provided'}
        """

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.3
        )

        translated_text = response.choices[0].message.content

        return {
            "original_text": text,
            "translated_text": translated_text,
            "context": context
        }