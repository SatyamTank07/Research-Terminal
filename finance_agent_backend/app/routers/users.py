from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db, get_single_user

router = APIRouter(prefix="/user", tags=["User"])


@router.get("/me")
def get_current_user(db: Session = Depends(get_db)):
    """Returns the profile of the single authorized chat user."""
    user = get_single_user(db)
    if not user:
        raise HTTPException(status_code=404, detail="Single user record not found or not initialized.")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Single user account is disabled.")
    return user.to_dict()
