from fastapi import APIRouter, HTTPException
from app.database import check_db_connection

router = APIRouter(tags=["Health"])


@router.get("/")
def root():
    return {"message": "Agent API is running. Visit /docs for Swagger UI."}


@router.get("/health")
def health():
    db_health = check_db_connection()
    return {
        "status": "healthy" if db_health.get("status") == "connected" else "degraded",
        "agent": "ready",
        "database": db_health,
    }


@router.get("/db-status")
def db_status():
    result = check_db_connection()
    if result.get("status") != "connected":
        raise HTTPException(status_code=503, detail=result)
    return result
