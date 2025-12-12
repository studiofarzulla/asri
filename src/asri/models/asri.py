"""ASRI database models."""

from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String

from .base import Base


class ASRIDaily(Base):
    """Daily ASRI calculation results."""

    __tablename__ = "asri_daily"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(DateTime, unique=True, nullable=False, index=True)

    # Aggregate ASRI
    asri = Column(Float, nullable=False)
    asri_normalized = Column(Float, nullable=False)
    asri_30d_avg = Column(Float, nullable=True)
    trend = Column(String(20), nullable=True)
    alert_level = Column(String(20), nullable=False)

    # Sub-indices
    stablecoin_risk = Column(Float, nullable=False)
    defi_liquidity_risk = Column(Float, nullable=False)
    contagion_risk = Column(Float, nullable=False)
    arbitrage_opacity = Column(Float, nullable=False)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<ASRIDaily(date={self.date}, asri={self.asri}, alert={self.alert_level})>"
