"""
Database Configuration and Models
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base

from app.config import DATABASE_URL

# Create async engine
engine = create_async_engine(DATABASE_URL, echo=False)

# Session factory
async_session = sessionmaker(
    engine, 
    class_=AsyncSession, 
    expire_on_commit=False
)

# Base class for models
Base = declarative_base()


class AlpacaCredentials(Base):
    """
    Stores encrypted Alpaca API credentials.
    Only one row should exist (single-user application).
    """
    __tablename__ = "alpaca_credentials"
    
    id = Column(Integer, primary_key=True, default=1)
    # Encrypted API key
    encrypted_api_key = Column(Text, nullable=False)
    # Encrypted API secret
    encrypted_api_secret = Column(Text, nullable=False)
    # Connection status
    is_connected = Column(Boolean, default=False)
    # Last successful connection verification
    last_verified_at = Column(DateTime, nullable=True)
    # When credentials were first stored
    created_at = Column(DateTime, default=datetime.utcnow)
    # When credentials were last updated
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AuditLog(Base):
    """
    Simple audit log for connection and refresh events.
    """
    __tablename__ = "audit_log"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    # Event type: 'connected', 'disconnected', 'refreshed', 'connection_failed'
    event_type = Column(String(50), nullable=False)
    # Additional details (no sensitive data)
    details = Column(Text, nullable=True)
    # When the event occurred
    created_at = Column(DateTime, default=datetime.utcnow)


async def init_db():
    """Initialize database tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncSession:
    """Get a database session."""
    async with async_session() as session:
        yield session
