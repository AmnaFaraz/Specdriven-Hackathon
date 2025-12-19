from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from core.database import get_db
from sqlalchemy.orm import Session
from core.security import get_current_user
from models.personalization import PersonalizationSettings
from typing import Optional

router = APIRouter()

# Request models
class PersonalizationUpdateRequest(BaseModel):
    user_id: str
    difficulty_level: str  # 'beginner', 'intermediate', 'advanced'
    content_focus: str     # 'theoretical', 'practical', 'application'
    language_preference: Optional[str] = "en"

class PersonalizationSettingsRequest(BaseModel):
    user_id: str
    personalization_settings: PersonalizationSettingsModel

class PersonalizationSettingsModel(BaseModel):
    difficulty_level: str  # 'beginner', 'intermediate', 'advanced'
    content_focus: str     # 'theoretical', 'practical', 'application'
    language_preference: Optional[str] = "en"

# Response models
class PersonalizationResponse(BaseModel):
    chapter_id: str
    difficulty_level: str
    content_focus: str
    language_preference: str

class PersonalizedContentResponse(BaseModel):
    original_content: str
    personalized_content: str
    applied_transformations: list

@router.get("/personalization/{chapter_id}", response_model=PersonalizationResponse)
async def get_personalization_settings(chapter_id: str, user_id: str, db: Session = Depends(get_db)):
    """
    Get personalization settings for a specific chapter.
    """
    setting = db.query(PersonalizationSettings).filter(
        PersonalizationSettings.chapter_id == chapter_id,
        PersonalizationSettings.user_id == int(user_id)
    ).first()

    if not setting:
        # Return default settings if none exist
        return PersonalizationResponse(
            chapter_id=chapter_id,
            difficulty_level="intermediate",
            content_focus="theoretical",
            language_preference="en"
        )

    return PersonalizationResponse(
        chapter_id=setting.chapter_id,
        difficulty_level=setting.difficulty_level,
        content_focus=setting.content_focus,
        language_preference=setting.language_preference
    )

@router.put("/personalization/{chapter_id}", status_code=200)
async def update_personalization_settings(
    chapter_id: str,
    request: PersonalizationUpdateRequest,
    db: Session = Depends(get_db)
):
    """
    Update personalization settings for a specific chapter.
    """
    # Check if setting already exists
    setting = db.query(PersonalizationSettings).filter(
        PersonalizationSettings.chapter_id == chapter_id,
        PersonalizationSettings.user_id == int(request.user_id)
    ).first()

    if setting:
        # Update existing setting
        setting.difficulty_level = request.difficulty_level
        setting.content_focus = request.content_focus
        setting.language_preference = request.language_preference
    else:
        # Create new setting
        setting = PersonalizationSettings(
            user_id=int(request.user_id),
            chapter_id=chapter_id,
            difficulty_level=request.difficulty_level,
            content_focus=request.content_focus,
            language_preference=request.language_preference
        )
        db.add(setting)

    db.commit()
    return {"message": "Personalization settings updated successfully"}

@router.post("/personalization/content/{chapter_id}", response_model=PersonalizedContentResponse)
async def get_personalized_content(
    chapter_id: str,
    request: PersonalizationSettingsRequest,
    db: Session = Depends(get_db)
):
    """
    Get personalized content for a chapter based on user preferences.
    """
    from services.personalization_service import PersonalizationService

    try:
        personalization_service = PersonalizationService(db)
        result = personalization_service.get_personalized_content(
            chapter_id=chapter_id,
            user_id=request.user_id,
            settings=request.personalization_settings
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))