"""
Credential Encryption Service

Handles encryption and decryption of Alpaca API credentials.
Uses Fernet symmetric encryption from the cryptography library.
"""
from cryptography.fernet import Fernet, InvalidToken
from app.config import ENCRYPTION_KEY


# Initialize Fernet cipher with the encryption key
_fernet = Fernet(ENCRYPTION_KEY.encode())


def encrypt_value(plaintext: str) -> str:
    """
    Encrypt a plaintext string.
    
    Args:
        plaintext: The string to encrypt
    
    Returns:
        Base64-encoded encrypted string
    """
    if not plaintext:
        raise ValueError("Cannot encrypt empty value")
    
    encrypted = _fernet.encrypt(plaintext.encode())
    return encrypted.decode()


def decrypt_value(ciphertext: str) -> str:
    """
    Decrypt an encrypted string.
    
    Args:
        ciphertext: Base64-encoded encrypted string
    
    Returns:
        Decrypted plaintext string
    
    Raises:
        ValueError: If decryption fails (invalid key or corrupted data)
    """
    if not ciphertext:
        raise ValueError("Cannot decrypt empty value")
    
    try:
        decrypted = _fernet.decrypt(ciphertext.encode())
        return decrypted.decode()
    except InvalidToken:
        raise ValueError("Failed to decrypt value - invalid key or corrupted data")
