from sqlalchemy import Column, Integer, String, DateTime, Text
from sqlalchemy.sql import func
from core.database import Base


class Chapter(Base):
    __tablename__ = "chapters"

    id = Column(Integer, primary_key=True, index=True)
    chapter_id = Column(String, unique=True, index=True, nullable=False)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    content_vector = Column(Text, nullable=True)  # Vector embedding as JSON string
    metadata = Column(Text, nullable=True)  # JSON string
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class TranslationCache(Base):
    __tablename__ = "translation_cache"

    id = Column(Integer, primary_key=True, index=True)
    original_content_id = Column(String, nullable=False)
    target_language = Column(String, nullable=False)
    translated_content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=False)


class SubagentExecutionLog(Base):
    __tablename__ = "subagent_execution_logs"

    id = Column(Integer, primary_key=True, index=True)
    subagent_name = Column(String, nullable=False)
    input_params = Column(Text, nullable=False)  # JSON string
    output_result = Column(Text, nullable=False)  # JSON string
    execution_time = Column(DateTime(timezone=True), server_default=func.now())
    user_id = Column(Integer, nullable=True)  # Optional foreign key to users