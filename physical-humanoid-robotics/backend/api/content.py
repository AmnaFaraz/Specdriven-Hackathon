from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from core.database import get_db
from sqlalchemy.orm import Session
from models.content import Chapter
import json

router = APIRouter()

# Request models
class ChapterCreateRequest(BaseModel):
    chapter_id: str
    title: str
    content: str
    metadata: Optional[Dict[str, Any]] = None

class ChapterUpdateRequest(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

# Response models
class ChapterResponse(BaseModel):
    id: int
    chapter_id: str
    title: str
    content: str
    content_vector: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: str
    updated_at: Optional[str] = None

@router.get("/chapters", response_model=List[ChapterResponse])
async def get_all_chapters(db: Session = Depends(get_db)):
    """
    Get all chapters.
    """
    chapters = db.query(Chapter).all()
    return [
        ChapterResponse(
            id=chapter.id,
            chapter_id=chapter.chapter_id,
            title=chapter.title,
            content=chapter.content,
            content_vector=chapter.content_vector,
            metadata=json.loads(chapter.metadata) if chapter.metadata else None,
            created_at=chapter.created_at.isoformat(),
            updated_at=chapter.updated_at.isoformat() if chapter.updated_at else None
        )
        for chapter in chapters
    ]

@router.get("/chapters/{chapter_id}", response_model=ChapterResponse)
async def get_chapter(chapter_id: str, db: Session = Depends(get_db)):
    """
    Get a specific chapter by chapter_id.
    """
    chapter = db.query(Chapter).filter(Chapter.chapter_id == chapter_id).first()
    if not chapter:
        raise HTTPException(status_code=404, detail="Chapter not found")

    return ChapterResponse(
        id=chapter.id,
        chapter_id=chapter.chapter_id,
        title=chapter.title,
        content=chapter.content,
        content_vector=chapter.content_vector,
        metadata=json.loads(chapter.metadata) if chapter.metadata else None,
        created_at=chapter.created_at.isoformat(),
        updated_at=chapter.updated_at.isoformat() if chapter.updated_at else None
    )

@router.post("/chapters", response_model=ChapterResponse, status_code=201)
async def create_chapter(request: ChapterCreateRequest, db: Session = Depends(get_db)):
    """
    Create a new chapter.
    """
    # Check if chapter already exists
    existing_chapter = db.query(Chapter).filter(Chapter.chapter_id == request.chapter_id).first()
    if existing_chapter:
        raise HTTPException(status_code=400, detail="Chapter with this ID already exists")

    chapter = Chapter(
        chapter_id=request.chapter_id,
        title=request.title,
        content=request.content,
        metadata=json.dumps(request.metadata) if request.metadata else None
    )

    db.add(chapter)
    db.commit()
    db.refresh(chapter)

    return ChapterResponse(
        id=chapter.id,
        chapter_id=chapter.chapter_id,
        title=chapter.title,
        content=chapter.content,
        content_vector=chapter.content_vector,
        metadata=json.loads(chapter.metadata) if chapter.metadata else None,
        created_at=chapter.created_at.isoformat(),
        updated_at=chapter.updated_at.isoformat() if chapter.updated_at else None
    )

@router.put("/chapters/{chapter_id}", response_model=ChapterResponse)
async def update_chapter(chapter_id: str, request: ChapterUpdateRequest, db: Session = Depends(get_db)):
    """
    Update a specific chapter.
    """
    chapter = db.query(Chapter).filter(Chapter.chapter_id == chapter_id).first()
    if not chapter:
        raise HTTPException(status_code=404, detail="Chapter not found")

    # Update fields that are provided
    if request.title is not None:
        chapter.title = request.title
    if request.content is not None:
        chapter.content = request.content
    if request.metadata is not None:
        chapter.metadata = json.dumps(request.metadata)

    db.commit()
    db.refresh(chapter)

    return ChapterResponse(
        id=chapter.id,
        chapter_id=chapter.chapter_id,
        title=chapter.title,
        content=chapter.content,
        content_vector=chapter.content_vector,
        metadata=json.loads(chapter.metadata) if chapter.metadata else None,
        created_at=chapter.created_at.isoformat(),
        updated_at=chapter.updated_at.isoformat() if chapter.updated_at else None
    )

@router.delete("/chapters/{chapter_id}", status_code=204)
async def delete_chapter(chapter_id: str, db: Session = Depends(get_db)):
    """
    Delete a specific chapter.
    """
    chapter = db.query(Chapter).filter(Chapter.chapter_id == chapter_id).first()
    if not chapter:
        raise HTTPException(status_code=404, detail="Chapter not found")

    db.delete(chapter)
    db.commit()

    return {"message": "Chapter deleted successfully"}