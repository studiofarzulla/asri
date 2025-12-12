# ASRI Development Setup Guide

## ✅ Already Done
- Repository cloned to: `/home/andrewroyce/asri`
- Python virtual environment created at: `.venv`
- All dependencies installed (FastAPI, PostgreSQL drivers, Pandas, etc.)
- `.env` file created from template

---

## 🚀 Quick Start

```bash
cd /home/andrewroyce/asri
source .venv/bin/activate
```

---

## 📋 What You Need Next

### 1. **PostgreSQL Database Setup**

#### Option A: Local PostgreSQL
```bash
# Install PostgreSQL
sudo apt update
sudo apt install postgresql postgresql-contrib

# Start PostgreSQL service
sudo systemctl start postgresql

# Create database and user
sudo -u postgres psql
CREATE DATABASE asri;
CREATE USER asri_user WITH ENCRYPTED PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE asri TO asri_user;
\q
```

#### Option B: Docker PostgreSQL (Recommended)
```bash
# Create docker-compose.yml in project root
docker-compose up -d

# Or run directly:
docker run -d \
  --name asri-postgres \
  -e POSTGRES_DB=asri \
  -e POSTGRES_USER=asri_user \
  -e POSTGRES_PASSWORD=secure_password \
  -p 5432:5432 \
  postgres:15
```

Update `.env` with your database connection:
```
DATABASE_URL=postgresql+asyncpg://asri_user:secure_password@localhost:5432/asri
DATABASE_SYNC_URL=postgresql://asri_user:secure_password@localhost:5432/asri
```

### 2. **API Keys Required**

Edit `.env` and add your API keys:

#### Free (Recommended to start):
- **FRED API Key** (Free): https://fred.stlouisfed.org/docs/api/api_key.html
  - For macro economic indicators
  
#### Paid/Premium (Optional):
- **Token Terminal**: https://tokenterminal.com/
  - Protocol metrics and revenue data
  
- **Messari API**: https://messari.io/api
  - On-chain data and crypto metrics
  
- **DeFi Llama**: https://defillama.com/docs/api
  - No key needed, but rate limited

---

## 🏗️ Components to Build

### Priority 1: Database Layer

#### Create Database Models (`src/asri/models/`)
```
src/asri/models/
├── __init__.py (already exists)
├── base.py          # SQLAlchemy Base setup
├── asri.py          # ASRI daily values table
├── sub_indices.py   # Sub-index values table
└── raw_data.py      # Raw data from sources
```

**Files to create:**
1. `base.py` - Database connection and base model
2. `asri.py` - Store daily ASRI calculations
3. `sub_indices.py` - Store sub-index components
4. `raw_data.py` - Store raw ingested data

#### Create Database Migrations
```bash
# Install alembic for migrations
pip install alembic

# Initialize alembic
alembic init alembic

# Create first migration
alembic revision --autogenerate -m "Initial tables"
alembic upgrade head
```

### Priority 2: Data Ingestion Layer

#### Expand Connectors (`src/asri/ingestion/`)
```
src/asri/ingestion/
├── __init__.py (already exists)
├── defillama.py (already exists)
├── token_terminal.py   # NEW - Token Terminal API
├── fred.py             # NEW - FRED macro data
├── messari.py          # NEW - Messari on-chain data
└── base.py             # NEW - Base connector class
```

**Files to create:**
1. `base.py` - Abstract base class for all connectors
2. `token_terminal.py` - Protocol metrics
3. `fred.py` - Economic indicators
4. `messari.py` - On-chain data

### Priority 3: Data Processing Pipeline

#### Create ETL Pipeline (`src/asri/pipeline/`)
```
src/asri/pipeline/
├── __init__.py
├── transform.py      # Raw data → sub-index inputs
├── calculate.py      # Compute sub-indices & ASRI
└── store.py          # Save to database
```

### Priority 4: Scheduler for Daily Updates

#### Create Scheduler (`src/asri/scheduler/`)
```
src/asri/scheduler/
├── __init__.py
├── jobs.py           # Define scheduled jobs
└── runner.py         # APScheduler configuration
```

**Daily job flow:**
1. Fetch data from all sources (6 AM UTC)
2. Transform raw data
3. Calculate sub-indices
4. Calculate ASRI
5. Store in database

### Priority 5: Connect API to Database

**Update** `src/asri/api/main.py`:
- Replace placeholder responses with database queries
- Add database session dependency
- Implement actual data retrieval

---

## 📁 Recommended File Structure

```
asri/
├── src/asri/
│   ├── api/
│   │   ├── main.py (✅ exists, needs DB connection)
│   │   ├── dependencies.py (NEW - DB session, auth)
│   │   └── schemas.py (NEW - Pydantic response models)
│   ├── ingestion/
│   │   ├── base.py (NEW)
│   │   ├── defillama.py (✅ exists)
│   │   ├── token_terminal.py (NEW)
│   │   ├── fred.py (NEW)
│   │   └── messari.py (NEW)
│   ├── models/
│   │   ├── base.py (NEW)
│   │   ├── asri.py (NEW)
│   │   ├── sub_indices.py (NEW)
│   │   └── raw_data.py (NEW)
│   ├── pipeline/
│   │   ├── __init__.py (NEW)
│   │   ├── transform.py (NEW)
│   │   ├── calculate.py (NEW)
│   │   └── store.py (NEW)
│   ├── scheduler/
│   │   ├── __init__.py (NEW)
│   │   ├── jobs.py (NEW)
│   │   └── runner.py (NEW)
│   ├── signals/
│   │   └── calculator.py (✅ exists)
│   └── config.py (NEW - Load .env settings)
├── alembic/ (NEW - DB migrations)
├── scripts/ (NEW - utility scripts)
│   ├── init_db.py
│   ├── backfill_data.py
│   └── test_sources.py
└── tests/
    ├── test_calculator.py (✅ exists)
    ├── test_ingestion.py (NEW)
    ├── test_models.py (NEW)
    └── test_pipeline.py (NEW)
```

---

## 🧪 Testing Your Setup

```bash
# Activate environment
cd /home/andrewroyce/asri
source .venv/bin/activate

# Run existing tests
pytest

# Start development server (will use placeholder data)
uvicorn asri.api.main:app --reload

# Visit in browser:
# http://localhost:8000/docs (Swagger UI)
# http://localhost:8000/health
# http://localhost:8000/asri/current
```

---

## 📝 Development Workflow

### Phase 1: Foundation (Week 1)
1. Set up PostgreSQL database
2. Create database models
3. Create and run migrations
4. Test database connectivity

### Phase 2: Data Ingestion (Week 2-3)
1. Get FRED API key (free)
2. Build additional data connectors
3. Test each connector independently
4. Create data validation logic

### Phase 3: Processing Pipeline (Week 3-4)
1. Build ETL pipeline
2. Transform raw data to sub-index inputs
3. Connect calculator to database
4. Validate calculations

### Phase 4: Scheduler & Automation (Week 4)
1. Set up APScheduler
2. Create daily update job
3. Add error handling and logging
4. Test full pipeline end-to-end

### Phase 5: API Integration (Week 5)
1. Connect API endpoints to database
2. Add query parameters and filters
3. Implement rate limiting
4. Add API authentication

### Phase 6: Polish (Week 6)
1. Add comprehensive tests
2. Write documentation
3. Set up CI/CD
4. Deploy to production

---

## 🔧 Useful Commands

```bash
# Activate environment
source .venv/bin/activate

# Run linter
ruff check src/

# Format code
ruff format src/

# Type checking
mypy src/

# Run tests with coverage
pytest --cov=src/asri --cov-report=html

# Start API server
uvicorn asri.api.main:app --reload --port 8000

# Run database migrations
alembic upgrade head

# Create new migration
alembic revision --autogenerate -m "description"
```

---

## 📚 Key Documentation Links

- **FastAPI**: https://fastapi.tiangolo.com/
- **SQLAlchemy**: https://docs.sqlalchemy.org/
- **Pydantic**: https://docs.pydantic.dev/
- **APScheduler**: https://apscheduler.readthedocs.io/
- **Alembic**: https://alembic.sqlalchemy.org/
- **DeFi Llama API**: https://defillama.com/docs/api
- **FRED API**: https://fred.stlouisfed.org/docs/api/

---

## 🆘 Troubleshooting

### Database Connection Issues
```bash
# Check if PostgreSQL is running
sudo systemctl status postgresql

# Test connection
psql -h localhost -U asri_user -d asri
```

### Import Errors
```bash
# Reinstall in editable mode
pip install -e ".[dev]"
```

### Port Already in Use
```bash
# Find process using port 8000
lsof -i :8000

# Use different port
uvicorn asri.api.main:app --reload --port 8001
```

---

## 🎯 Next Immediate Steps

1. **Set up PostgreSQL** (see section 1 above)
2. **Get FRED API key** (free, takes 2 minutes)
3. **Create database models** (start with `src/asri/models/base.py`)
4. **Test API** (`uvicorn asri.api.main:app --reload`)

The foundation is ready - you can start building! 🚀
