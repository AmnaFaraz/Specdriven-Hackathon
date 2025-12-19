from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv

load_dotenv()

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://username:password@localhost:5432/physical_humanoid_robotics")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Qdrant configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    prefer_grpc=False  # Using HTTP instead of gRPC for simplicity
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_vector_db():
    """Initialize the Qdrant collection for book embeddings"""
    from qdrant_client.http import models

    # Define the collection for storing book embeddings
    qdrant_client.recreate_collection(
        collection_name="book_embeddings",
        vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),  # Using OpenAI embedding size
    )

    print("Qdrant collection 'book_embeddings' created successfully")