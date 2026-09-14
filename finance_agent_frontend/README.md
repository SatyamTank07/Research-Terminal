# FinanceAgent UI (React + TypeScript + Vite + Bun + shadcn/ui)

A modern, responsive financial AI conversational user interface built with:
- **Runtime & Package Manager**: [Bun](https://bun.sh/)
- **Build Tool**: [Vite](https://vite.dev/)
- **Frontend Framework**: [React 19](https://react.dev/) + [TypeScript](https://www.typescriptlang.org/)
- **Styling**: [Tailwind CSS v4](https://tailwindcss.com/)
- **UI Components**: [shadcn/ui](https://ui.shadcn.com/) primitives (Card, Button, Badge, Input, Textarea, Avatar)
- **Icons**: [Lucide React](https://lucide.dev/)

---

## 🚀 Quickstart

### 1. Install Dependencies
```bash
bun install
```

### 2. Start Development Server
```bash
bun run dev
```

The application will start on **http://localhost:5173**.

### 3. Ensure Backend is Running
The backend should be running on port 8000:
```bash
cd ../finance_agent_backend
# Either with Docker:
docker compose up
# Or directly with Uvicorn:
uvicorn main:app --reload --port 8000
```

---

## 🛠️ Configuration
By default, the frontend connects to `http://localhost:8000`. You can configure a custom URL by copying `.env.example` to `.env`:
```env
VITE_API_BASE_URL=http://localhost:8000
```

---

## 📦 Build for Production
```bash
bun run build
bun run preview
```
