from typing import List, Dict, Any
import google.generativeai as genai
from core.config import settings
from core.database import qdrant_client
import json


class GeminiService:
    def __init__(self):
        # Configure the API key
        genai.configure(api_key=settings.gemini_api_key)
        
        # Initialize the model
        self.model = genai.GenerativeModel('gemini-pro')
        
        # Initialize Qdrant client
        self.qdrant_client = qdrant_client

    def generate_response(self, prompt: str) -> str:
        """
        Generate a response using the Gemini model.
        """
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.7,
                    "max_output_tokens": 1000,
                }
            )
            return response.text
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Sorry, I encountered an error while processing your request."

    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for text using Gemini's embedding model.
        """
        try:
            result = genai.embed_content(
                model="models/embedding-001",
                content=[text],
                task_type="retrieval_document"
            )
            return result['embedding'][0]
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return []

    def query_rag(self, query: str, context: str, selected_text: str = None, user_id: str = None):
        """
        Query the RAG system using Gemini to get answers about the book content with source citations.
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
        embedding = self.get_embedding(search_text)

        if not embedding:
            return {
                "response": "Error: Could not generate embeddings for the query.",
                "sources": []
            }

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

        # Generate response using Gemini
        prompt = f"""
        Based on the following context, answer the question: {query}

        Context: {context_text}
        """

        answer = self.generate_response(prompt)

        return {
            "response": answer,
            "sources": sources
        }