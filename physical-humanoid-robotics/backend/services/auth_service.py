from typing import Optional
from sqlalchemy.orm import Session
from core.security import verify_password, get_password_hash
from models.user import User
from datetime import datetime, timedelta
from core.config import settings
import jwt

class AuthService:
    def __init__(self, db: Session):
        self.db = db

    def authenticate_user(self, email: str, password: str) -> Optional[User]:
        """
        Authenticate a user with email and password.
        """
        user = self.db.query(User).filter(User.email == email).first()
        if not user or not verify_password(password, user.password_hash):
            return None
        return user

    def create_access_token(self, user_id: int) -> str:
        """
        Create an access token for the user.
        """
        from core.security import create_access_token

        data = {"sub": str(user_id)}
        token = create_access_token(data=data)
        return token

    def get_current_user(self, token: str) -> Optional[User]:
        """
        Get the current user from the token.
        """
        from core.security import verify_token

        try:
            payload = verify_token(token)
            user_id: str = payload.get("sub")
            if user_id is None:
                return None
            user = self.db.query(User).filter(User.id == int(user_id)).first()
            return user
        except jwt.PyJWTError:
            return None

    def register_user(self, email: str, name: str, password: str,
                     software_background: str = None, hardware_background: str = None) -> User:
        """
        Register a new user with the provided information.
        """
        # Check if user already exists
        existing_user = self.db.query(User).filter(User.email == email).first()
        if existing_user:
            raise ValueError("User with this email already exists")

        # Create new user
        hashed_password = get_password_hash(password)
        db_user = User(
            email=email,
            name=name,
            password_hash=hashed_password,
            software_background=software_background,
            hardware_background=hardware_background
        )

        self.db.add(db_user)
        self.db.commit()
        self.db.refresh(db_user)

        return db_user