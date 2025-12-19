from typing import Dict, Any
from sqlalchemy.orm import Session
from core.config import settings
from openai import OpenAI
from models.user import User
from models.personalization import PersonalizationSettings
from pydantic import BaseModel

class PersonalizationSettingsModel(BaseModel):
    difficulty_level: str  # 'beginner', 'intermediate', 'advanced'
    content_focus: str     # 'theoretical', 'practical', 'application'
    language_preference: str = "en"

class PersonalizationService:
    def __init__(self, db: Session):
        self.db = db
        self.client = OpenAI(api_key=settings.openai_api_key)

    def get_personalized_content(self, chapter_id: str, user_id: str,
                                settings: PersonalizationSettingsModel) -> Dict[str, Any]:
        """
        Get personalized content for a chapter based on user preferences.
        """
        # In a real implementation, we would fetch the original chapter content
        # For now, using a placeholder
        original_content = f"This is the original content for chapter {chapter_id}"

        # Apply personalization based on settings
        personalized_content = self._apply_personalization(
            original_content,
            settings.difficulty_level,
            settings.content_focus
        )

        return {
            "original_content": original_content,
            "personalized_content": personalized_content,
            "applied_transformations": [
                f"Adjusted for {settings.difficulty_level} level",
                f"Focusing on {settings.content_focus} aspects"
            ]
        }

    def _apply_personalization(self, content: str, difficulty_level: str, content_focus: str) -> str:
        """
        Apply personalization transformations to content based on user preferences.
        """
        # Prepare prompt for personalization
        prompt = f"""
        Adapt the following content based on these parameters:
        - Difficulty level: {difficulty_level}
        - Content focus: {content_focus}

        Original content:
        {content}

        Return a version that is appropriate for the specified difficulty level and focus area.
        """

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.6
        )

        return response.choices[0].message.content

    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """
        Get the user's preferences for personalization.
        """
        user = self.db.query(User).filter(User.id == int(user_id)).first()
        if not user:
            return {}

        # Get the user's personalization settings
        settings = self.db.query(PersonalizationSettings).filter(
            PersonalizationSettings.user_id == int(user_id)
        ).all()

        preferences = {
            "email": user.email,
            "name": user.name,
            "software_background": user.software_background,
            "hardware_background": user.hardware_background,
            "chapter_preferences": []
        }

        for setting in settings:
            preferences["chapter_preferences"].append({
                "chapter_id": setting.chapter_id,
                "difficulty_level": setting.difficulty_level,
                "content_focus": setting.content_focus,
                "language_preference": setting.language_preference
            })

        return preferences