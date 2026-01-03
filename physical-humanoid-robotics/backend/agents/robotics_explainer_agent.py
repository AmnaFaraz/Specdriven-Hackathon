from typing import Dict, Any
from core.config import settings
from services.gemini_service import GeminiService

class RoboticsExplainerAgent:
    def __init__(self):
        self.gemini_service = GeminiService()

    def explain_concept(self, concept: str, user_background: str = None) -> Dict[str, Any]:
        """
        Explain a robotics concept in detail based on user's background.
        """
        background_context = f"The user has background in: {user_background}" if user_background else "The user has general background"
        prompt = f"""
        {background_context}

        Explain the following robotics concept in detail:
        {concept}

        Provide a comprehensive explanation that includes:
        1. Definition and core principles
        2. Practical applications
        3. Relevant examples
        4. Technical details appropriate to the user's background
        """

        explanation = self.gemini_service.generate_response(prompt)

        return {
            "concept": concept,
            "explanation": explanation,
            "background_applied": user_background is not None
        }