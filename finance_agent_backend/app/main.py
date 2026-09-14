import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import app.agents  # Ensures agents self-register with AgentRegistry
from app.database import init_db
from app.routers import chat, conversations, documents, health, users

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("finance_agent")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing database and seeding default single user...")
    try:
        init_db()
    except Exception as e:
        logger.error(f"Error initializing database tables/seeding: {e}")
    yield


app = FastAPI(title="LangChain Agent API", lifespan=lifespan)

# Add CORS Middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(health.router)
app.include_router(users.router)
app.include_router(conversations.router)
app.include_router(chat.router)
app.include_router(documents.router)
