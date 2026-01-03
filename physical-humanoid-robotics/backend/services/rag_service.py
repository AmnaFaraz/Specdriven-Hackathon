from typing import List, Dict, Any
from sqlalchemy.orm import Session
from core.database import qdrant_client
from core.config import settings
from services.gemini_service import GeminiService
from datetime import datetime
import json

class RAGService:
    def __init__(self, db: Session):
        self.db = db
        self.qdrant_client = qdrant_client
        self.gemini_service = GeminiService()

    def query(self, query: str, context: str, selected_text: str = None, user_id: str = None):
        """
        Query the RAG system to get answers about the book content with source citations.
        """
        # Use the Gemini service to handle the entire RAG process
        result = self.gemini_service.query_rag(query, context, selected_text, user_id)

        # Save to history if user_id is provided
        if user_id:
            self._save_to_history(user_id, query, result["response"], result["sources"])

        return result

    def get_history(self, user_id: str, session_id: str):
        """
        Get chat history for a specific user and session.
        """
        # In a real implementation, this would query the database
        # For now, returning empty list
        return []

    def _save_to_history(self, user_id: str, query: str, response: str, sources: List[Dict]):
        """
        Save the interaction to history.
        """
        # In a real implementation, this would save to the database
        # For now, just logging
        print(f"Saving to history - User: {user_id}, Query: {query[:50]}...")