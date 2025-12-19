from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.sql import func
from core.database import Base


class PersonalizationSettings(Base):
    __tablename__ = "personalization_settings"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    chapter_id = Column(String, nullable=False)
    difficulty_level = Column(String, nullable=False)  # 'beginner', 'intermediate', 'advanced'
    content_focus = Column(String, nullable=False)  # 'theoretical', 'practical', 'application'
    language_preference = Column(String, default="en", nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())