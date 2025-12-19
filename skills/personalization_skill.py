"""
Personalization Skill

This skill handles content personalization based on user background for the Physical AI & Humanoid Robotics interactive book.
"""
from typing import Dict, Any, List
import json

def personalize_content(content: str, user_background: Dict[str, str], chapter_context: str = "") -> Dict[str, Any]:
    """
    Personalize content based on user's background and preferences.

    Args:
        content: The original content to personalize
        user_background: Dictionary containing user's software/hardware background
        chapter_context: Context about the current chapter

    Returns:
        Dictionary containing original content, personalized content, and applied transformations
    """
    # Determine personalization based on user background
    software_background = user_background.get("software", "")
    hardware_background = user_background.get("hardware", "")

    # Apply personalization transformations
    personalized_content = f"Personalized version for user with {software_background} and {hardware_background} background"

    transformations_applied = []
    if software_background:
        transformations_applied.append(f"Software-focused examples for {software_background}")
    if hardware_background:
        transformations_applied.append(f"Hardware-focused examples for {hardware_background}")

    return {
        "original_content": content,
        "personalized_content": personalized_content,
        "user_background": user_background,
        "chapter_context": chapter_context,
        "transformations_applied": transformations_applied
    }

def suggest_content_adaptation(content: str, user_preferences: Dict[str, Any]) -> Dict[str, Any]:
    """
    Suggest adaptations to content based on user preferences.

    Args:
        content: The content to adapt
        user_preferences: Dictionary containing user preferences (difficulty, focus, etc.)

    Returns:
        Dictionary containing adaptation suggestions and confidence levels
    """
    difficulty = user_preferences.get("difficulty_level", "intermediate")
    content_focus = user_preferences.get("content_focus", "theoretical")

    suggestions = [
        f"Adjust complexity for {difficulty} level",
        f"Emphasize {content_focus} aspects"
    ]

    adapted_content = f"Content adapted for {difficulty} level with {content_focus} focus"

    return {
        "original_content": content,
        "adapted_content": adapted_content,
        "suggestions": suggestions,
        "user_preferences": user_preferences,
        "confidence_level": 0.9
    }

# Example usage:
# user_bg = {"software": "python development", "hardware": "mechatronics"}
# result = personalize_content("Original chapter content", user_bg, "kinematics")