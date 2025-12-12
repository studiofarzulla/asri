# ASRI Development Checklist ✅

## Setup Complete ✅

- [x] Repository cloned to `/home/andrewroyce/asri`
- [x] Python virtual environment created
- [x] All dependencies installed (64 packages)
- [x] `.env` file created
- [x] Docker Compose file for PostgreSQL
- [x] Documentation created

---

## Your First Hour (Start Here!)

### ☐ 1. Activate Environment (30 seconds)
```bash
cd /home/andrewroyce/asri
source .venv/bin/activate
```

### ☐ 2. Test Current System (2 minutes)
```bash
# Run tests
pytest

# Start API server (Ctrl+C to stop)
uvicorn asri.api.main:app --reload
```
**Expected:** Server starts, visit http://localhost:8000/docs

### ☐ 3. Start PostgreSQL (2 minutes)
```bash
# Make sure Docker is running, then:
docker-compose up -d
```
**Expected:** PostgreSQL running on port 5432

### ☐ 4. Get FRED API Key (5 minutes)
1. Visit: https://fred.stlouisfed.org/docs/api/api_key.html
2. Create free account
3. Copy your API key
4. Edit `.env` and paste: `FRED_API_KEY=your_key_here`

---

## Your First Day (Core Setup)

### ☐ 5. Create Database Models (30 minutes)

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
from sqlalchemy import Column, Integer, Float, DateTime, String
from .base import Base

class ASRIDaily(Base):
    __tablename__ = "asri_daily"
    
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, unique=True, nullable=False, index=True)
    asri = Column(Float, nullable=False)
    asri_normalized = Column(Float, nullable=False)
    alert_level = Column(String(20), nullable=False)
    
    stablecoin_risk = Column(Float, nullable=False)
    defi_liquidity_risk = Column(Float, nullable=False)
    contagion_risk = Column(Float, nullable=False)
    arbitrage_opacity = Column(Float, nullable=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
```

### ☐ 6. Set Up Database Migrations (10 minutes)
```bash
# Install Alembic
pip install alembic

# Initialize
alembic init alembic

# Edit alembic.ini - set sqlalchemy.url to your DATABASE_URL
# Or better, edit alembic/env.py to read from .env

# Create first migration
alembic revision --autogenerate -m "Initial schema"

# Apply migration
alembic upgrade head
```

### ☐ 7. Test Database Connection (5 minutes)
Create `scripts/test_db.py`:
```python
import asyncio
from src.asri.models.base import engine
from src.asri.models.asri import ASRIDaily, Base

async def test_connection():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("✅ Database connection successful!")

if __name__ == "__main__":
    asyncio.run(test_connection())
```

Run: `python scripts/test_db.py`

---

## Your First Week (Core Features)

### ☐ 8. Build FRED Connector (2 hours)

Create `src/asri/ingestion/fred.py`:
```python
import httpx
from typing import Dict, Any
from datetime import datetime

class FREDConnector:
    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = httpx.AsyncClient()
    
    async def fetch_series(self, series_id: str, start_date: str) -> Dict[str, Any]:
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": start_date,
        }
        response = await self.client.get(self.BASE_URL, params=params)
        response.raise_for_status()
        return response.json()
    
    async def get_treasury_rates(self, start_date: str):
        # DGS10 = 10-Year Treasury Rate
        return await self.fetch_series("DGS10", start_date)
    
    async def get_vix(self, start_date: str):
        # VIXCLS = CBOE VIX
        return await self.fetch_series("VIXCLS", start_date)
```

Test it:
```python
import asyncio
from src.asri.ingestion.fred import FREDConnector
from dotenv import load_dotenv
import os

load_dotenv()

async def test():
    fred = FREDConnector(os.getenv("FRED_API_KEY"))
    data = await fred.get_treasury_rates("2024-01-01")
    print(data)

asyncio.run(test())
```

### ☐ 9. Create Data Pipeline (3 hours)

Create `src/asri/pipeline/orchestrator.py`:
```python
from datetime import datetime
from src.asri.ingestion.fred import FREDConnector
from src.asri.ingestion.defillama import DefiLlamaConnector
from src.asri.signals.calculator import compute_asri
from src.asri.models.asri import ASRIDaily
from src.asri.models.base import async_session

async def run_daily_calculation():
    """Run the complete ASRI calculation pipeline."""
    
    # 1. Fetch data from sources
    fred = FREDConnector(api_key=...)
    defillama = DefiLlamaConnector()
    
    treasury_data = await fred.get_treasury_rates(...)
    defi_data = await defillama.get_protocol_tvls(...)
    
    # 2. Transform raw data (implement this)
    # transformed = transform_data(treasury_data, defi_data)
    
    # 3. Calculate ASRI
    # result = compute_asri(...)
    
    # 4. Store in database
    async with async_session() as session:
        daily_record = ASRIDaily(
            date=datetime.utcnow(),
            asri=result.asri,
            asri_normalized=result.asri_normalized,
            alert_level=result.alert_level,
            stablecoin_risk=result.sub_indices.stablecoin_risk,
            # ... other fields
        )
        session.add(daily_record)
        await session.commit()
```

### ☐ 10. Wire Up Scheduler (1 hour)

Create `src/asri/scheduler/jobs.py`:
```python
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from src.asri.pipeline.orchestrator import run_daily_calculation

scheduler = AsyncIOScheduler()

def start_scheduler():
    # Run daily at 1 AM UTC
    scheduler.add_job(
        run_daily_calculation,
        'cron',
        hour=1,
        minute=0,
        id='daily_asri_calculation'
    )
    scheduler.start()
    
def stop_scheduler():
    scheduler.shutdown()
```

Update `src/asri/api/main.py` lifespan:
```python
from src.asri.scheduler.jobs import start_scheduler, stop_scheduler

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Starting ASRI API v{__version__}")
    start_scheduler()  # Add this
    yield
    print("Shutting down ASRI API")
    stop_scheduler()  # Add this
```

### ☐ 11. Connect API to Database (2 hours)

Update `src/asri/api/main.py`:
```python
from src.asri.models.base import async_session
from src.asri.models.asri import ASRIDaily
from sqlalchemy import select, desc

@app.get("/asri/current", response_model=ASRIResponse)
async def get_current_asri():
    async with async_session() as session:
        stmt = select(ASRIDaily).order_by(desc(ASRIDaily.date)).limit(1)
        result = await session.execute(stmt)
        record = result.scalar_one_or_none()
        
        if not record:
            raise HTTPException(status_code=404, detail="No ASRI data available")
        
        return ASRIResponse(
            timestamp=record.date,
            asri=record.asri,
            asri_30d_avg=record.asri,  # TODO: Calculate actual 30-day avg
            trend="rising",  # TODO: Calculate trend
            sub_indices=SubIndices(
                stablecoin_risk=record.stablecoin_risk,
                defi_liquidity_risk=record.defi_liquidity_risk,
                contagion_risk=record.contagion_risk,
                arbitrage_opacity=record.arbitrage_opacity,
            ),
            alert_level=record.alert_level,
            last_update=record.created_at,
        )
```

---

## Testing Checklist

### ☐ Unit Tests
- [ ] Test FRED connector with mock responses
- [ ] Test ASRI calculator with known inputs
- [ ] Test database models CRUD operations
- [ ] Test data transformations

### ☐ Integration Tests
- [ ] Test full pipeline with real API calls
- [ ] Test API endpoints with database
- [ ] Test scheduler job execution

### ☐ Code Quality
- [ ] Run `ruff check src/` - no errors
- [ ] Run `mypy src/` - no type errors
- [ ] Run `pytest --cov=src/asri` - >80% coverage
- [ ] All docstrings added

---

## Launch Checklist

### ☐ Pre-Production
- [ ] All API keys secured in environment
- [ ] Database backups configured
- [ ] Logging configured (structlog)
- [ ] Error monitoring (Sentry)
- [ ] Rate limiting implemented
- [ ] CORS properly configured

### ☐ Deployment
- [ ] Dockerfile created
- [ ] Docker image built and tested
- [ ] Production database provisioned
- [ ] SSL certificate configured
- [ ] Domain name configured
- [ ] CI/CD pipeline set up

### ☐ Documentation
- [ ] API documentation complete
- [ ] User guide written
- [ ] Developer guide written
- [ ] README updated

---

## Progress Tracker

**Foundation:** ✅✅✅✅✅ 100% (Setup complete!)

**Week 1:** ⬜⬜⬜⬜⬜ 0% (Database models, migrations)

**Week 2:** ⬜⬜⬜⬜⬜ 0% (Data connectors)

**Week 3:** ⬜⬜⬜⬜⬜ 0% (Pipeline & calculator)

**Week 4:** ⬜⬜⬜⬜⬜ 0% (Scheduler & automation)

**Week 5:** ⬜⬜⬜⬜⬜ 0% (API integration)

**Week 6:** ⬜⬜⬜⬜⬜ 0% (Testing & polish)

**Overall:** ████⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜ 15%

---

## Need Help?

📖 Read `SETUP_GUIDE.md` - comprehensive documentation
⚡ Read `QUICK_START.md` - quick reference
📋 Read `TODO.md` - detailed task breakdown
📁 Read `docs/ASRI-PROPOSAL.md` - full methodology

🚀 **You've got everything you need. Time to build!**
