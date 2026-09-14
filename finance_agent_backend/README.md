# Simple LangChain Agent with FastAPI, Uvicorn & PostgreSQL in Docker

A backend service calling an OpenAI LLM using LangChain's `create_agent` harness with **FastAPI**, **Uvicorn** (hot-reloading enabled), and **PostgreSQL 16** managed via Docker Compose.

---

## 📁 Structure

- [`app/main.py`](app/main.py): FastAPI application entrypoint, lifespan event, CORS configuration, and router setup.
- [`app/routers/`](app/routers/): Modular HTTP endpoints (`chat.py`, `conversations.py`, `health.py`, `users.py`).
- [`app/services/`](app/services/): Business orchestration logic (`chat_service.py`).
- [`app/agents/`](app/agents/): Extensible multi-agent framework (`BaseAgent`, `AgentRegistry`, specialized agents, prompts, and tools).
- [`app/schemas/`](app/schemas/): Pydantic validation schemas.
- [`app/models.py`](app/models.py): SQLAlchemy models (`User`, `Conversation`, `ChatMessage`).
- [`app/database.py`](app/database.py): Engine, session generator (`get_db`), table initialization & default user seeding (`init_db`).
- [`app/config.py`](app/config.py): Centralized environment settings.
- [`docker-compose.yml`](docker-compose.yml): Multi-container definition for `finance-agent` (FastAPI) and `postgres`.
- [`Dockerfile`](Dockerfile): Python 3.11 environment.
- [`requirements.txt`](requirements.txt): Dependencies.
- [`.env.example`](.env.example): Environment variable template.

---

## ?? Quickstart

### 1. Configure Environment Variables
Ensure `.env` inside `finance_agent_backend` contains:
```env
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...

POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=finance_agent_db
POSTGRES_PORT=5432
POSTGRES_HOST=postgres
DATABASE_URL=postgresql+psycopg2://postgres:postgres@postgres:5432/finance_agent_db

DEFAULT_USER_USERNAME=finance_user
DEFAULT_USER_EMAIL=user@financeagent.local
DEFAULT_USER_FULL_NAME=Finance User
```

### 2. Run with Docker Compose
From inside `finance_agent_backend/`:
```bash
docker compose up --build
```
This will:
1. Start the `finance_postgres` container and initialize the database.
2. Wait for the PostgreSQL healthcheck to report healthy.
3. Launch `finance_agent`, create the `users` table, and automatically seed the default single user.

### 3. Verify Endpoints

- **Swagger UI**: Open **[http://localhost:8000/docs](http://localhost:8000/docs)**
- **Overall Health Check**:
  ```bash
  curl http://localhost:8000/health
  ```
- **Database Status**:
  ```bash
  curl http://localhost:8000/db-status
  ```
- **Single User Profile**:
  ```bash
  curl http://localhost:8000/user/me
  ```
- **Chat Endpoint** (Protected by single user check):
  ```bash
  curl -X POST http://localhost:8000/chat \
    -H "Content-Type: application/json" \
    -d '{"message": "Hello! Give me a one-sentence financial tip."}'
  ```

---

## ?? Hot-Reloading & Persistence
- Any edit to Python source code in `finance_agent_backend` automatically triggers hot-reload without rebuilding the container.
- PostgreSQL data is persisted across container restarts in the Docker named volume `postgres_data`.
