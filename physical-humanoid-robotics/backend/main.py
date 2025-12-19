from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from core.config import settings
from core.database import engine, Base

# Import all models to ensure they are registered with Base before creating tables
from models import user as user_model, chat as chat_model, personalization as personalization_model, content as content_model

from api import rag, user, personalization, translation, history, subagent, content

# Create database tables
Base.metadata.create_all(bind=engine)

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    debug=settings.debug,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(rag.router, prefix="/api", tags=["rag"])
app.include_router(user.router, prefix="/api", tags=["user"])
app.include_router(personalization.router, prefix="/api", tags=["personalization"])
app.include_router(translation.router, prefix="/api", tags=["translation"])
app.include_router(history.router, prefix="/api", tags=["history"])
app.include_router(subagent.router, prefix="/api", tags=["subagent"])
app.include_router(content.router, prefix="/api", tags=["content"])

@app.get("/")
def read_root():
    return {"message": "Physical AI & Humanoid Robotics Interactive Book API"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "version": settings.version}