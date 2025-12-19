from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from core.database import get_db
from sqlalchemy.orm import Session
from typing import Optional

router = APIRouter()

# Request models
class TranslationRequest(BaseModel):
    text: str
    target_language: str = "ur"  # Default to Urdu
    source_language: Optional[str] = "en"
    context: Optional[str] = None  # Context for translation (e.g., chapter title)
    user_id: Optional[str] = None

class TranslationChapterRequest(BaseModel):
    target_language: str = "ur"
    user_id: str

# Response models
class TranslationResponse(BaseModel):
    original_text: str
    translated_text: str
    detected_source_language: str
    target_language: str
    cached: bool

class ChapterTranslationResponse(BaseModel):
    chapter_id: str
    original_content: str
    translated_content: str
    language: str

@router.post("/translate", response_model=TranslationResponse)
async def translate_content(request: TranslationRequest, db: Session = Depends(get_db)):
    """
    Translate content to the specified language.
    """
    from services.translation_service import TranslationService

    try:
        translation_service = TranslationService(db)
        result = translation_service.translate(
            text=request.text,
            target_language=request.target_language,
            source_language=request.source_language,
            context=request.context,
            user_id=request.user_id
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/translate/chapter/{chapter_id}", response_model=ChapterTranslationResponse)
async def get_translated_chapter(
    chapter_id: str,
    target_language: str = "ur",
    user_id: str = None,
    db: Session = Depends(get_db)
):
    """
    Get a translated version of a chapter.
    """
    from services.translation_service import TranslationService

    try:
        translation_service = TranslationService(db)
        result = translation_service.get_translated_chapter(
            chapter_id=chapter_id,
            target_language=target_language,
            user_id=user_id
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))