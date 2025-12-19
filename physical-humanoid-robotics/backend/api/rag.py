from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from core.database import get_db
from sqlalchemy.orm import Session
from core.security import get_current_user
from services.rag_service import RAGService
from models.user import User

router = APIRouter()

# Request models
class RAGQueryRequest(BaseModel):
    query: str
    context: str  # 'entire_book', 'current_chapter', 'selected_text'
    selected_text: Optional[str] = None
    user_id: Optional[str] = None

class RAGQueryResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]]

# Response models
class ChatHistoryItem(BaseModel):
    query: str
    response: str
    timestamp: str

@router.post("/rag/query", response_model=RAGQueryResponse)
async def query_rag(request: RAGQueryRequest, db: Session = Depends(get_db)):
    """
    Query the RAG system to get answers about the book content with source citations.
    """
    try:
        rag_service = RAGService(db)
        result = rag_service.query(
            query=request.query,
            context=request.context,
            selected_text=request.selected_text,
            user_id=request.user_id
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/rag/history", response_model=List[ChatHistoryItem])
async def get_chat_history(user_id: str, session_id: str, db: Session = Depends(get_db)):
    """
    Get chat history for a specific user and session.
    """
    try:
        rag_service = RAGService(db)
        history = rag_service.get_history(user_id, session_id)
        return history
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))