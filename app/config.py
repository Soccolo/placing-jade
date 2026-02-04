"""
Notrix Application Configuration
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Database
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./notrix.db")

# Encryption key for credentials (REQUIRED)
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")
if not ENCRYPTION_KEY:
    key_path_env = os.getenv("ENCRYPTION_KEY_PATH")
    candidate_paths = []
    if key_path_env:
        candidate_paths.append(Path(key_path_env))
    else:
        candidate_paths.extend(
            [
                Path("/data/encryption_key"),
                BASE_DIR / ".encryption_key",
            ]
        )

    for candidate in candidate_paths:
        if candidate.exists():
            ENCRYPTION_KEY = candidate.read_text().strip()
            break

    if not ENCRYPTION_KEY:
        try:
            from cryptography.fernet import Fernet

            ENCRYPTION_KEY = Fernet.generate_key().decode()
            for candidate in candidate_paths:
                try:
                    candidate.parent.mkdir(parents=True, exist_ok=True)
                    candidate.write_text(ENCRYPTION_KEY)
                    break
                except Exception:
                    continue
        except Exception as e:
            raise ValueError(
                "ENCRYPTION_KEY environment variable is required. "
                "Generate one with: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
            ) from e

if not ENCRYPTION_KEY:
    raise ValueError(
        "ENCRYPTION_KEY environment variable is required. "
        "Generate one with: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
    )

# Target portfolio path
TARGET_PORTFOLIO_PATH = os.getenv(
    "TARGET_PORTFOLIO_PATH", 
    str(BASE_DIR / "data" / "target_portfolio.csv")
)

# Validation settings
try:
    WEIGHT_SUM_EPSILON = float(os.getenv("WEIGHT_SUM_EPSILON", "0.01"))
except (ValueError, TypeError) as e:
    env_value = os.getenv("WEIGHT_SUM_EPSILON")
    raise ValueError(
        f"Invalid WEIGHT_SUM_EPSILON environment variable: '{env_value}'. "
        f"Must be a numeric value (e.g., '0.01'). Error: {e}"
    )
