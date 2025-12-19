# Quickstart: Physical AI & Humanoid Robotics Interactive Book

## Prerequisites
- Python 3.11+
- Node.js 18+
- Git
- Access to Neon Serverless Postgres
- Access to Qdrant Cloud
- OpenAI API key
- Claude Code access

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Backend Setup
```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration
```

### 3. Frontend Setup
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration
```

### 4. Database Setup
```bash
# With virtual environment activated
cd backend

# Run database migrations
python -m alembic upgrade head
```

### 5. Vector Database Setup
```bash
# Initialize Qdrant collection for embeddings
python -c "
from backend.src.core.database import init_vector_db
init_vector_db()
"
```

### 6. Running the Application

#### Development Mode:
```bash
# Terminal 1: Start backend
cd backend
python -m uvicorn src.main:app --reload --port 8000

# Terminal 2: Start frontend
cd frontend
npm run start
```

#### Production Mode:
```bash
# Build frontend
cd frontend
npm run build

# Start backend with production server
cd backend
gunicorn src.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## Key Endpoints

### Backend API
- `POST /api/auth/register` - User registration
- `POST /api/auth/login` - User login
- `POST /api/rag/query` - RAG chatbot query
- `GET /api/personalization/{chapter_id}` - Get personalization for chapter
- `PUT /api/personalization/{chapter_id}` - Update personalization for chapter
- `POST /api/translate` - Translate content to Urdu

### Frontend
- `http://localhost:3000` - Main book interface
- Chatbot embedded in each chapter page
- Personalization toggle per chapter
- Translation toggle per chapter

## Environment Variables

### Backend (.env)
```
DATABASE_URL=postgresql://user:password@localhost:5432/book_db
QDRANT_URL=https://your-cluster.qdrant.tech:6333
QDRANT_API_KEY=your_api_key
OPENAI_API_KEY=your_openai_key
CLAUDE_API_KEY=your_claude_key
SECRET_KEY=your_secret_key
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

### Frontend (.env)
```
REACT_APP_API_BASE_URL=http://localhost:8000
REACT_APP_CLAUDE_API_KEY=your_claude_key
```

## Testing
```bash
# Backend tests
cd backend
python -m pytest tests/

# Frontend tests
cd frontend
npm run test
```

## Deployment
1. Configure Vercel for frontend deployment
2. Configure Vercel or similar for backend deployment
3. Set environment variables in deployment platform
4. Ensure Neon Postgres and Qdrant Cloud are properly configured