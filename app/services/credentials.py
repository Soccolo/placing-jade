"""
Credentials Service

Manages storage and retrieval of encrypted Alpaca credentials.
"""
from datetime import datetime
from typing import Optional
from sqlalchemy import select, delete

from app.database import async_session, AlpacaCredentials, AuditLog
from app.services.encryption import encrypt_value, decrypt_value


class DecryptedCredentials:
    """Decrypted credentials for use in Alpaca API calls."""
    def __init__(self, api_key: str, api_secret: str, is_connected: bool, last_verified_at: Optional[datetime]):
        self.api_key = api_key
        self.api_secret = api_secret
        self.is_connected = is_connected
        self.last_verified_at = last_verified_at


async def save_credentials(api_key: str, api_secret: str) -> None:
    """
    Save encrypted Alpaca credentials to the database.
    Replaces any existing credentials.
    
    Args:
        api_key: Alpaca API key (plaintext)
        api_secret: Alpaca API secret (plaintext)
    """
    # Encrypt credentials
    encrypted_key = encrypt_value(api_key)
    encrypted_secret = encrypt_value(api_secret)
    
    async with async_session() as session:
        # Delete any existing credentials
        await session.execute(delete(AlpacaCredentials))
        
        # Create new credentials record
        creds = AlpacaCredentials(
            id=1,
            encrypted_api_key=encrypted_key,
            encrypted_api_secret=encrypted_secret,
            is_connected=False,
            last_verified_at=None
        )
        session.add(creds)
        await session.commit()


async def get_credentials() -> Optional[DecryptedCredentials]:
    """
    Get decrypted Alpaca credentials from the database.
    
    Returns:
        DecryptedCredentials if credentials exist, None otherwise
    """
    async with async_session() as session:
        result = await session.execute(
            select(AlpacaCredentials).where(AlpacaCredentials.id == 1)
        )
        creds = result.scalar_one_or_none()
        
        if creds is None:
            return None
        
        try:
            api_key = decrypt_value(creds.encrypted_api_key)
            api_secret = decrypt_value(creds.encrypted_api_secret)
            
            return DecryptedCredentials(
                api_key=api_key,
                api_secret=api_secret,
                is_connected=creds.is_connected,
                last_verified_at=creds.last_verified_at
            )
        except ValueError:
            # Decryption failed - corrupted data or wrong key
            return None


async def update_connection_status(is_connected: bool, verified_at: Optional[datetime] = None) -> None:
    """
    Update the connection status in the database.
    
    Args:
        is_connected: Whether the connection was successful
        verified_at: When the connection was verified (defaults to now if connected)
    """
    async with async_session() as session:
        result = await session.execute(
            select(AlpacaCredentials).where(AlpacaCredentials.id == 1)
        )
        creds = result.scalar_one_or_none()
        
        if creds:
            creds.is_connected = is_connected
            if is_connected:
                creds.last_verified_at = verified_at or datetime.utcnow()
            await session.commit()


async def delete_credentials() -> None:
    """Delete all stored credentials."""
    async with async_session() as session:
        await session.execute(delete(AlpacaCredentials))
        await session.commit()


async def log_audit_event(event_type: str, details: Optional[str] = None) -> None:
    """
    Log an audit event to the database.
    
    Args:
        event_type: Type of event (connected, disconnected, refreshed, connection_failed)
        details: Additional details (no sensitive data)
    """
    async with async_session() as session:
        event = AuditLog(
            event_type=event_type,
            details=details
        )
        session.add(event)
        await session.commit()


async def get_recent_audit_events(limit: int = 10) -> list:
    """
    Get recent audit events.
    
    Args:
        limit: Maximum number of events to return
    
    Returns:
        List of AuditLog entries, most recent first
    """
    async with async_session() as session:
        result = await session.execute(
            select(AuditLog)
            .order_by(AuditLog.created_at.desc())
            .limit(limit)
        )
        return result.scalars().all()
