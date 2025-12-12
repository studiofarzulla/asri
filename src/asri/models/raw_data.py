"""Raw data storage models."""

from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, JSON, String, Text

from .base import Base


class RawDataSource(Base):
    """Store raw data from external sources."""

    __tablename__ = "raw_data_sources"

    id = Column(Integer, primary_key=True, index=True)
    source = Column(String(50), nullable=False, index=True)
    data_type = Column(String(50), nullable=False)
    fetch_date = Column(DateTime, nullable=False, index=True)

    # Store raw JSON response
    raw_data = Column(JSON, nullable=False)

    # Optional: Store any errors
    error = Column(Text, nullable=True)
    status = Column(String(20), default="success")

    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<RawDataSource(source={self.source}, type={self.data_type}, date={self.fetch_date})>"
