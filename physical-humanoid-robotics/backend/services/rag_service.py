from typing import List, Dict, Any
from sqlalchemy.orm import Session
from core.database import qdrant_client
from core.config import settings
from openai import OpenAI
from datetime import datetime
import json

class RAGService:
    def __init__(self, db: Session):
        self.db = db
        self.qdrant_client = qdrant_client
        self.client = OpenAI(api_key=settings.openai_api_key)

    def query(self, query: str, context: str, selected_text: str = None, user_id: str = None):
        """
        Query the RAG system to get answers about the book content with source citations.
        """
        # Determine search context
        if context == "selected_text" and selected_text:
            search_text = selected_text
        elif context == "current_chapter":
            # In a real implementation, we would have the current chapter content
            search_text = query
        else:  # entire_book
            search_text = query

        # Generate embedding for the query
        embedding = self._get_embedding(search_text)

        # Search in Qdrant
        search_results = self.qdrant_client.search(
            collection_name="book_embeddings",
            query_vector=embedding,
            limit=5,  # Return top 5 results
        )

        # Prepare context from search results
        context_text = ""
        sources = []
        for result in search_results:
            context_text += result.payload.get("text_excerpt", "") + "\n"
            sources.append({
                "chapter_id": result.payload.get("chapter_id", ""),
                "title": result.payload.get("title", ""),
                "content": result.payload.get("text_excerpt", ""),
                "similarity_score": result.score
            })

        # Generate response using OpenAI
        prompt = f"""
        Based on the following context, answer the question: {query}

        Context: {context_text}
        """

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7
        )

        answer = response.choices[0].message.content

        # Save to history if user_id is provided
        if user_id:
            self._save_to_history(user_id, query, answer, sources)

        return {
            "response": answer,
            "sources": sources
        }

    def get_history(self, user_id: str, session_id: str):
        """
        Get chat history for a specific user and session.
        """
        # In a real implementation, this would query the database
        # For now, returning empty list
        return []

    def _get_embedding(self, text: str):
        """
        Get embedding for text using OpenAI.
        """
        response = self.client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding

    def _save_to_history(self, user_id: str, query: str, response: str, sources: List[Dict]):
        """
        Save the interaction to history.
        """
        # In a real implementation, this would save to the database
        # For now, just logging
        print(f"Saving to history - User: {user_id}, Query: {query[:50]}...")