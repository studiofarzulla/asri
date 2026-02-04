-- ASRI D1 Database Schema

CREATE TABLE IF NOT EXISTS asri_daily (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL UNIQUE,
    asri REAL NOT NULL,
    asri_30d_avg REAL,
    trend TEXT,
    alert_level TEXT NOT NULL,
    stablecoin_risk REAL NOT NULL,
    defi_liquidity_risk REAL NOT NULL,
    contagion_risk REAL NOT NULL,
    arbitrage_opacity REAL NOT NULL,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_asri_date ON asri_daily(date);
CREATE INDEX IF NOT EXISTS idx_asri_alert ON asri_daily(alert_level);

-- Insert some seed data for demo
INSERT OR IGNORE INTO asri_daily (date, asri, asri_30d_avg, trend, alert_level, stablecoin_risk, defi_liquidity_risk, contagion_risk, arbitrage_opacity)
VALUES
    ('2024-12-01', 41.2, 40.5, 'stable', 'moderate', 37.8, 43.5, 46.2, 34.1),
    ('2024-12-02', 42.1, 40.8, 'rising', 'moderate', 38.2, 44.1, 47.0, 34.8),
    ('2024-12-03', 41.8, 40.9, 'stable', 'moderate', 38.0, 43.8, 46.5, 34.5),
    ('2024-12-04', 43.5, 41.2, 'rising', 'moderate', 39.1, 45.2, 48.1, 35.2),
    ('2024-12-05', 42.7, 41.5, 'rising', 'moderate', 38.5, 44.5, 47.5, 35.0);
