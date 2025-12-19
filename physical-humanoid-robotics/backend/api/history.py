from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from core.database import get_db
from sqlalchemy.orm import Session
from models.chat import ChatHistory
from typing import List
from datetime import datetime

router = APIRouter()

# Request models
class SaveHistoryRequest(BaseModel):
    user_id: int
    session_id: str
    query: str
    response: str
    source_citations: str = None
    chapter_context: str = None

# Response models
class HistoryResponse(BaseModel):
    id: int
    user_id: int
    session_id: str
    query: str
    response: str
    source_citations: str = None
    created_at: str
    chapter_context: str = None

@router.post("/history/save", status_code=201)
async def save_chat_history(request: SaveHistoryRequest, db: Session = Depends(get_db)):
    """
    Save a chat interaction to the history.
    """
    try:
        chat_history = ChatHistory(
            user_id=request.user_id,
            session_id=request.session_id,
            query=request.query,
            response=request.response,
            source_citations=request.source_citations,
            chapter_context=request.chapter_context
        )

        db.add(chat_history)
        db.commit()
        db.refresh(chat_history)

        return {"message": "Chat history saved successfully", "id": chat_history.id}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history/fetch", response_model=List[HistoryResponse])
async def fetch_chat_history(user_id: int, session_id: str, db: Session = Depends(get_db)):
    """
    Fetch chat history for a specific user and session.
    """
    try:
        history_items = db.query(ChatHistory).filter(
            ChatHistory.user_id == user_id,
            ChatHistory.session_id == session_id
        ).order_by(ChatHistory.created_at.desc()).all()

        return [
            HistoryResponse(
                id=item.id,
                user_id=item.user_id,
                session_id=item.session_id,
                query=item.query,
                response=item.response,
                source_citations=item.source_citations,
                created_at=item.created_at.isoformat(),
                chapter_context=item.chapter_context
            )
            for item in history_items
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))