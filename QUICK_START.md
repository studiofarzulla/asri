# ASRI Quick Start ⚡

## Current Status ✅
- ✅ Repository cloned
- ✅ Virtual environment created
- ✅ All dependencies installed
- ✅ API server tested and working
- ✅ Environment file created

## Start Development NOW

```bash
cd /home/andrewroyce/asri
source .venv/bin/activate
```

## Test Current API (Works Now!)

```bash
# Start server
uvicorn asri.api.main:app --reload

# Open browser to:
# http://localhost:8000/docs (Interactive API documentation)
```

## Database Setup (5 minutes)

### Option 1: Docker (Easiest)
```bash
# Start PostgreSQL + pgAdmin
docker-compose up -d

# Database is now running at localhost:5432
# pgAdmin at http://localhost:5050
```

### Option 2: Local Install
```bash
sudo apt install postgresql
sudo -u postgres psql -c "CREATE DATABASE asri;"
sudo -u postgres psql -c "CREATE USER asri_user WITH PASSWORD 'password';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE asri TO asri_user;"
```

## Update .env File

Edit `/home/andrewroyce/asri/.env`:
```bash
# Use this for Docker setup:
DATABASE_URL=postgresql+asyncpg://asri_user:asri_dev_password@localhost:5432/asri
DATABASE_SYNC_URL=postgresql://asri_user:asri_dev_password@localhost:5432/asri

# Get free API key from https://fred.stlouisfed.org/docs/api/api_key.html
FRED_API_KEY=your_key_here

# Others are optional for now
```

## What Works Right Now

✅ API server starts
✅ Swagger docs at /docs
✅ Health check endpoint
✅ Calculator logic for ASRI
✅ DeFi Llama connector skeleton
✅ Test suite

## What Needs Building

### Immediate (This Week):
1. **Database models** - Define tables for storing ASRI data
2. **Database migrations** - Set up Alembic
3. **Connect API to DB** - Replace placeholder data

### Short Term (Next 2 Weeks):
4. **Data connectors** - Token Terminal, FRED, Messari
5. **ETL pipeline** - Transform raw data → ASRI
6. **Scheduler** - Daily automated calculations

### Polish (Month 2):
7. **Rate limiting** - API access control
8. **Authentication** - Premium tier
9. **Frontend** - Dashboard UI
10. **Deployment** - Production hosting

## Project Structure Overview

```
asri/
├── .env                  ✅ Environment config
├── .venv/               ✅ Python virtual environment
├── docker-compose.yml   ✅ PostgreSQL setup
├── SETUP_GUIDE.md       ✅ Full documentation
├── QUICK_START.md       ✅ This file!
│
├── src/asri/
│   ├── api/             ✅ FastAPI app (working!)
│   │   └── main.py
│   ├── ingestion/       🔨 Data connectors (1/4 done)
│   │   └── defillama.py
│   ├── signals/         ✅ ASRI calculator
│   │   └── calculator.py
│   ├── models/          ⚠️  Database models (NEED TO BUILD)
│   │   └── __init__.py
│   └── __init__.py
│
├── tests/               ✅ Test suite
│   └── test_calculator.py
│
└── docs/
    └── ASRI-PROPOSAL.md ✅ Full methodology
```

## Key Files to Read

1. `README.md` - Project overview
2. `SETUP_GUIDE.md` - Detailed setup instructions
3. `docs/ASRI-PROPOSAL.md` - Full technical specification
4. `src/asri/api/main.py` - API endpoints (see TODOs)
5. `src/asri/signals/calculator.py` - Risk calculation logic

## Test the Current System

```bash
# Run tests
pytest

# Check code quality
ruff check src/

# Type checking
mypy src/

# Start API server
uvicorn asri.api.main:app --reload
```

## Get API Keys (Free!)

### 1. FRED API (Macro Economic Data)
- Visit: https://fred.stlouisfed.org/docs/api/api_key.html
- Sign up (free)
- Get API key
- Add to `.env`: `FRED_API_KEY=your_key`

### 2. DeFi Llama (No Key Needed!)
- Public API: https://defillama.com/docs/api
- Rate limited but free
- Already has connector started

## First Development Task

### Create Database Models (30 minutes)

Create `src/asri/models/base.py`:
```python
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from pydantic_settings import BaseSettings

Base = declarative_base()

class Settings(BaseSettings):
    database_url: str
    
    class Config:
        env_file = ".env"

settings = Settings()
engine = create_async_engine(settings.database_url, echo=True)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
```

Create `src/asri/models/asri.py`:
```python
from datetime import datetime
from sqlalchemy import Column, Integer, Float, DateTime
from .base import Base

class ASRIDaily(Base):
    __tablename__ = "asri_daily"
    
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, unique=True, nullable=False)
    asri = Column(Float, nullable=False)
    asri_normalized = Column(Float, nullable=False)
    alert_level = Column(String(20), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
```

## Resources

- **Project Repository**: https://github.com/studiofarzulla/asri
- **FastAPI Docs**: https://fastapi.tiangolo.com/
- **SQLAlchemy Docs**: https://docs.sqlalchemy.org/
- **Full Setup Guide**: `SETUP_GUIDE.md`

## Support

Everything is set up and ready! Start with:
1. Run `docker-compose up -d` (start database)
2. Create database models (see SETUP_GUIDE.md)
3. Build your first data connector

🚀 **You're ready to build!**
