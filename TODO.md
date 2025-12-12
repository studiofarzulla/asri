# ASRI Development Roadmap & TODO List

## ✅ COMPLETED
- [x] Repository structure
- [x] Core dependencies installed
- [x] FastAPI application skeleton
- [x] ASRI calculation logic
- [x] DeFi Llama connector started
- [x] Basic test suite
- [x] API documentation (Swagger)
- [x] Development environment setup

---

## 🚀 PHASE 1: Database Foundation (Priority: HIGH)

### Week 1: Database Setup

- [ ] **1.1 Install Database Dependencies**
  - [ ] Install Alembic: `pip install alembic`
  - [ ] Initialize Alembic: `alembic init alembic`
  - [ ] Configure alembic.ini with DATABASE_URL

- [ ] **1.2 Create Database Models**
  - [ ] Create `src/asri/models/base.py`
    - Database connection setup
    - Async session configuration
    - Base model class
  - [ ] Create `src/asri/models/asri.py`
    - ASRIDaily table (date, asri, normalized, alert_level)
    - ASRIHistory for backtest data
  - [ ] Create `src/asri/models/sub_indices.py`
    - SubIndicesDaily table
    - Store all 4 sub-index values per day
  - [ ] Create `src/asri/models/raw_data.py`
    - RawDataSource table (source, timestamp, data)
    - For storing raw API responses

- [ ] **1.3 Database Migrations**
  - [ ] Create initial migration: `alembic revision --autogenerate -m "initial schema"`
  - [ ] Review generated migration
  - [ ] Apply migration: `alembic upgrade head`
  - [ ] Test database connection

- [ ] **1.4 Configuration Management**
  - [ ] Create `src/asri/config.py`
    - Load settings from .env
    - Database connection strings
    - API keys management
  - [ ] Update .env with database credentials
  - [ ] Test configuration loading

---

## 🔌 PHASE 2: Data Ingestion (Priority: HIGH)

### Week 2: Build Data Connectors

- [ ] **2.1 Base Connector Class**
  - [ ] Create `src/asri/ingestion/base.py`
    - Abstract base class
    - Rate limiting logic
    - Error handling
    - Retry mechanism
  
- [ ] **2.2 FRED Connector** (EASIEST - Start Here!)
  - [ ] Get free FRED API key
  - [ ] Create `src/asri/ingestion/fred.py`
    - Fetch Treasury rates
    - Fetch VIX data
    - Fetch GDP indicators
  - [ ] Write tests for FRED connector
  - [ ] Test with real API calls

- [ ] **2.3 Finish DeFi Llama Connector**
  - [ ] Complete `src/asri/ingestion/defillama.py`
    - TVL data for top protocols
    - Stablecoin market caps
    - Historical data fetching
  - [ ] Add error handling
  - [ ] Write tests

- [ ] **2.4 Token Terminal Connector** (Requires paid API)
  - [ ] Get Token Terminal API key
  - [ ] Create `src/asri/ingestion/token_terminal.py`
    - Protocol revenue data
    - Active users
    - TVL metrics
  - [ ] Write tests

- [ ] **2.5 Messari Connector** (Requires paid API)
  - [ ] Get Messari API key
  - [ ] Create `src/asri/ingestion/messari.py`
    - On-chain metrics
    - Asset profiles
    - Market data
  - [ ] Write tests

- [ ] **2.6 Integration Tests**
  - [ ] Test all connectors together
  - [ ] Validate data formats
  - [ ] Check rate limiting
  - [ ] Error handling validation

---

## ⚙️ PHASE 3: Data Processing Pipeline (Priority: HIGH)

### Week 3: ETL Pipeline

- [ ] **3.1 Create Pipeline Module**
  - [ ] Create `src/asri/pipeline/__init__.py`
  - [ ] Create `src/asri/pipeline/transform.py`
    - Transform raw DeFi Llama data
    - Transform FRED economic data
    - Normalize all inputs to 0-100 scale
  
- [ ] **3.2 Connect Calculator to Pipeline**
  - [ ] Update `src/asri/signals/calculator.py`
    - Accept transformed data as input
    - Add data validation
    - Handle missing data gracefully
  
- [ ] **3.3 Data Storage Layer**
  - [ ] Create `src/asri/pipeline/store.py`
    - Save raw data to database
    - Save sub-index calculations
    - Save final ASRI values
    - Transaction handling

- [ ] **3.4 Pipeline Orchestration**
  - [ ] Create `src/asri/pipeline/orchestrator.py`
    - Coordinate: fetch → transform → calculate → store
    - Error handling for each stage
    - Logging and monitoring
  
- [ ] **3.5 Testing**
  - [ ] Create `tests/test_pipeline.py`
  - [ ] Test with mock data
  - [ ] Test with real API data
  - [ ] Test error scenarios

---

## ⏰ PHASE 4: Scheduling & Automation (Priority: MEDIUM)

### Week 4: Daily Updates

- [ ] **4.1 Create Scheduler Module**
  - [ ] Create `src/asri/scheduler/__init__.py`
  - [ ] Create `src/asri/scheduler/jobs.py`
    - Daily ASRI calculation job
    - Data source health checks
    - Cleanup old data job

- [ ] **4.2 APScheduler Integration**
  - [ ] Create `src/asri/scheduler/runner.py`
    - Initialize scheduler
    - Configure job triggers (daily at 1 AM UTC)
    - Start/stop scheduler
  
- [ ] **4.3 Update API Lifespan**
  - [ ] Modify `src/asri/api/main.py`
    - Start scheduler on startup
    - Stop scheduler on shutdown
    - Add scheduler status endpoint

- [ ] **4.4 Monitoring & Alerts**
  - [ ] Add logging to all scheduled jobs
  - [ ] Email alerts on failure (optional)
  - [ ] Job execution history tracking

---

## 🔗 PHASE 5: API Integration (Priority: MEDIUM)

### Week 5: Connect API to Database

- [ ] **5.1 Database Dependencies**
  - [ ] Create `src/asri/api/dependencies.py`
    - Database session dependency
    - Authentication dependency (future)
    - Rate limiting dependency (future)

- [ ] **5.2 Update API Endpoints**
  - [ ] `/asri/current` - Query latest ASRI from database
  - [ ] `/asri/timeseries` - Query historical data with date range
  - [ ] `/asri/subindex/{name}` - Query specific sub-index history
  - [ ] `/asri/stress-test` - Implement actual stress test logic
  - [ ] `/asri/methodology` - Return from database or static file

- [ ] **5.3 Query Optimization**
  - [ ] Add database indexes
  - [ ] Optimize date range queries
  - [ ] Add caching layer (Redis optional)

- [ ] **5.4 Response Models**
  - [ ] Create `src/asri/api/schemas.py`
    - Pydantic models for all responses
    - Validation logic
    - Example responses

---

## 🧪 PHASE 6: Testing & Quality (Priority: HIGH)

### Week 6: Comprehensive Testing

- [ ] **6.1 Unit Tests**
  - [ ] Test all data connectors
  - [ ] Test calculator functions
  - [ ] Test data transformations
  - [ ] Test database models

- [ ] **6.2 Integration Tests**
  - [ ] Test full pipeline end-to-end
  - [ ] Test API endpoints with database
  - [ ] Test scheduler jobs

- [ ] **6.3 Code Quality**
  - [ ] Fix all Ruff linting errors
  - [ ] Fix all MyPy type errors
  - [ ] Add docstrings to all functions
  - [ ] Achieve >80% test coverage

- [ ] **6.4 Performance Testing**
  - [ ] Load test API endpoints
  - [ ] Optimize slow database queries
  - [ ] Profile memory usage

---

## 📈 PHASE 7: Advanced Features (Priority: LOW)

### Future Enhancements

- [ ] **7.1 Rate Limiting**
  - [ ] Implement rate limiting middleware
  - [ ] Create free tier (100 req/day)
  - [ ] Create pro tier (5000 req/day)
  - [ ] Add API key authentication

- [ ] **7.2 Stress Testing**
  - [ ] Implement scenario engine
  - [ ] Add predefined scenarios (treasury spike, DeFi crash)
  - [ ] Allow custom scenarios
  - [ ] Store stress test results

- [ ] **7.3 Historical Backtesting**
  - [ ] Backfill data from 2020-01-01
  - [ ] Validate against known crises
  - [ ] Generate backtest reports
  - [ ] Compare with other risk indices

- [ ] **7.4 Web Dashboard**
  - [ ] Create React/Vue frontend
  - [ ] Real-time ASRI chart
  - [ ] Sub-index breakdown
  - [ ] Historical trends
  - [ ] Alert notifications

- [ ] **7.5 Alerts & Notifications**
  - [ ] Email alerts on high risk
  - [ ] Webhook notifications
  - [ ] Telegram/Discord bot
  - [ ] SMS alerts (Twilio)

---

## 🚀 DEPLOYMENT (Priority: MEDIUM)

### Production Readiness

- [ ] **8.1 Containerization**
  - [ ] Create production Dockerfile
  - [ ] Optimize image size
  - [ ] Multi-stage build
  - [ ] Health checks

- [ ] **8.2 CI/CD Pipeline**
  - [ ] GitHub Actions workflow
  - [ ] Automated testing
  - [ ] Automated deployment
  - [ ] Version tagging

- [ ] **8.3 Infrastructure**
  - [ ] Set up production database (AWS RDS / DigitalOcean)
  - [ ] Set up application server (AWS EC2 / DigitalOcean)
  - [ ] Configure DNS
  - [ ] SSL certificates (Let's Encrypt)

- [ ] **8.4 Monitoring**
  - [ ] Application monitoring (Sentry)
  - [ ] Database monitoring
  - [ ] Uptime monitoring
  - [ ] Log aggregation

- [ ] **8.5 Documentation**
  - [ ] API documentation (OpenAPI/Swagger)
  - [ ] User guide
  - [ ] Developer setup guide
  - [ ] Deployment guide

---

## 📝 DOCUMENTATION

- [ ] API reference documentation
- [ ] Architecture diagrams
- [ ] Data flow diagrams
- [ ] Contributing guidelines
- [ ] Change log / Release notes

---

## 🎯 QUICK WINS (Do These First!)

1. **Set up PostgreSQL** - 5 minutes with Docker
2. **Get FRED API key** - 2 minutes, completely free
3. **Create database models** - 30 minutes
4. **Build FRED connector** - 1 hour
5. **Test end-to-end** - Connect FRED → transform → calculate → store

Start with these and you'll have working data flowing through the system!

---

## 📊 Progress Tracking

- **Phase 1**: ⬜⬜⬜⬜⬜ 0% (Database)
- **Phase 2**: ⬜⬜⬜⬜⬜ 0% (Data Ingestion)
- **Phase 3**: ⬜⬜⬜⬜⬜ 0% (Pipeline)
- **Phase 4**: ⬜⬜⬜⬜⬜ 0% (Scheduling)
- **Phase 5**: ⬜⬜⬜⬜⬜ 0% (API Integration)
- **Phase 6**: ⬜⬜⬜⬜⬜ 0% (Testing)
- **Overall**: 15% Complete

*Update this as you complete tasks!*
