from typing import Dict, Any
from core.config import settings
from openai import OpenAI

class PersonalizationAgent:
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)

    def suggest_content_adaptation(self, content: str, user_preferences: Dict[str, Any]) -> Dict[str, Any]:
        """
        Suggest adaptations to content based on user preferences.
        """
        difficulty_level = user_preferences.get("difficulty_level", "intermediate")
        content_focus = user_preferences.get("content_focus", "theoretical")
        background = user_preferences.get("background", "")

        prompt = f"""
        Adapt the following content based on user preferences:
        - Difficulty level: {difficulty_level}
        - Content focus: {content_focus}
        - User background: {background}

        Original content:
        {content}

        Provide a personalized version that matches the user's level and interests.
        """

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.6
        )

        personalized_content = response.choices[0].message.content

        return {
            "original_content": content,
            "personalized_content": personalized_content,
            "applied_preferences": user_preferences
        }