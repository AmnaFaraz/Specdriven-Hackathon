from typing import Dict, Any, Optional
from sqlalchemy.orm import Session
from core.config import settings
from openai import OpenAI
from models.content import TranslationCache
from datetime import datetime, timedelta
import json

class TranslationService:
    def __init__(self, db: Session):
        self.db = db
        self.client = OpenAI(api_key=settings.openai_api_key)

    def translate(self, text: str, target_language: str = "ur",
                  source_language: str = "en", context: str = None,
                  user_id: str = None) -> Dict[str, Any]:
        """
        Translate content to the specified language.
        """
        # Check if translation is cached
        cache_key = f"{source_language}:{target_language}:{hash(text[:50])}"
        cached_translation = self._get_cached_translation(cache_key)

        if cached_translation:
            return {
                "original_text": text,
                "translated_text": cached_translation.translated_content,
                "detected_source_language": source_language,
                "target_language": target_language,
                "cached": True
            }

        # Perform translation using OpenAI
        prompt = f"""
        Translate the following text to {target_language}:
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

        # Cache the translation
        self._cache_translation(cache_key, text, translated_text, target_language, user_id)

        return {
            "original_text": text,
            "translated_text": translated_text,
            "detected_source_language": source_language,
            "target_language": target_language,
            "cached": False
        }

    def get_translated_chapter(self, chapter_id: str, target_language: str = "ur",
                              user_id: str = None) -> Dict[str, Any]:
        """
        Get a translated version of a chapter.
        """
        # In a real implementation, this would fetch the chapter content from the database
        # For now, returning a placeholder
        original_content = f"This is the original content of chapter {chapter_id}"

        # Translate the chapter content
        translation_result = self.translate(
            text=original_content,
            target_language=target_language,
            user_id=user_id
        )

        return {
            "chapter_id": chapter_id,
            "original_content": original_content,
            "translated_content": translation_result["translated_text"],
            "language": target_language
        }

    def _get_cached_translation(self, cache_key: str) -> Optional[TranslationCache]:
        """
        Get cached translation if it exists and hasn't expired.
        """
        cached = self.db.query(TranslationCache).filter(
            TranslationCache.original_content_id == cache_key
        ).first()

        if cached and datetime.utcnow() < cached.expires_at:
            return cached
        return None

    def _cache_translation(self, cache_key: str, original_text: str,
                          translated_text: str, target_language: str,
                          user_id: str = None):
        """
        Cache a translation with expiration.
        """
        # Remove any existing cache for this key
        self.db.query(TranslationCache).filter(
            TranslationCache.original_content_id == cache_key
        ).delete()

        # Create new cache entry (expires in 24 hours)
        cache_entry = TranslationCache(
            original_content_id=cache_key,
            target_language=target_language,
            translated_content=translated_text,
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(hours=24)
        )

        self.db.add(cache_entry)
        self.db.commit()