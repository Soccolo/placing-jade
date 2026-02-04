import os


def get_api_keys():
    """
    Load API keys from environment variables.
    Returns a dict with provider keys, using empty strings as defaults.
    
    Environment variables:
    - FINNHUB_API_KEY
    - TWELVE_DATA_API_KEY
    - ALPHA_VANTAGE_API_KEY
    - FMP_API_KEY
    """
    api_keys = {
        "finnhub": os.getenv("FINNHUB_API_KEY", ""),
        "twelve_data": os.getenv("TWELVE_DATA_API_KEY", ""),
        "alpha_vantage": os.getenv("ALPHA_VANTAGE_API_KEY", ""),
        "fmp": os.getenv("FMP_API_KEY", ""),
    }
    
    # Log warning if required keys are missing
    if not api_keys["finnhub"]:
        print("WARNING: FINNHUB_API_KEY environment variable is not set")
    
    return api_keys
