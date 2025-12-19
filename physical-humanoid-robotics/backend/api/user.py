from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel
from core.database import get_db
from sqlalchemy.orm import Session
from core.security import get_password_hash
from models.user import User
from typing import Optional

router = APIRouter()

# Request models
class UserCreateRequest(BaseModel):
    email: str
    name: str
    password: str
    software_background: Optional[str] = None
    hardware_background: Optional[str] = None

class UserLoginRequest(BaseModel):
    email: str
    password: str

class UserResponse(BaseModel):
    id: int
    email: str
    name: str

# Response models
class TokenResponse(BaseModel):
    access_token: str
    token_type: str

@router.post("/auth/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(user_data: UserCreateRequest, db: Session = Depends(get_db)):
    """
    Register a new user with background information.
    """
    # Check if user already exists
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this email already exists"
        )

    # Create new user
    hashed_password = get_password_hash(user_data.password)
    db_user = User(
        email=user_data.email,
        name=user_data.name,
        password_hash=hashed_password,
        software_background=user_data.software_background,
        hardware_background=user_data.hardware_background
    )

    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    return UserResponse(
        id=db_user.id,
        email=db_user.email,
        name=db_user.name
    )

@router.post("/auth/login", response_model=TokenResponse)
async def login_user(user_data: UserLoginRequest, db: Session = Depends(get_db)):
    """
    Login user and return access token.
    """
    from core.security import verify_password
    from core.config import settings
    from datetime import timedelta
    from core.security import create_access_token

    # Find user by email
    user = db.query(User).filter(User.email == user_data.email).first()
    if not user or not verify_password(user_data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create access token
    access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
    access_token = create_access_token(
        data={"sub": str(user.id)}, expires_delta=access_token_expires
    )

    return TokenResponse(access_token=access_token, token_type="bearer")