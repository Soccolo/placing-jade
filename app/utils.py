"""
Utility functions for the Notrix application.
"""
from urllib.parse import urlparse


# Define allowed hosts for redirects (relative paths are always allowed)
ALLOWED_HOSTS = {
    "localhost",
    "127.0.0.1",
    "[::1]",  # IPv6 localhost
}

# Dangerous URL schemes that should be blocked
DANGEROUS_SCHEMES = {
    "javascript",
    "data",
    "vbscript",
    "file",
    "about",
}


def is_safe_redirect_url(url: str, allowed_hosts: set = ALLOWED_HOSTS) -> bool:
    """
    Validate that a redirect URL is safe and not malicious.
    
    Args:
        url: The URL to validate
        allowed_hosts: Set of allowed hostnames (for absolute URLs)
    
    Returns:
        True if the URL is safe, False otherwise
    
    Safe URLs are:
    - Relative paths (starting with /)
    - Absolute URLs with http/https scheme to allowed hosts only
    
    Unsafe URLs include:
    - javascript:, data:, vbscript:, file: schemes
    - Absolute URLs to hosts not in the allowlist
    - URLs with unusual characters or encoding attempts
    """
    if not url or not isinstance(url, str):
        return False
    
    url = url.strip()
    
    # Empty or whitespace-only URLs are not safe
    if not url:
        return False
    
    # Parse the URL
    parsed = urlparse(url)
    
    # Check for dangerous schemes
    if parsed.scheme and parsed.scheme.lower() in DANGEROUS_SCHEMES:
        return False
    
    # Relative paths are safe (no scheme, no netloc)
    if not parsed.scheme and not parsed.netloc:
        # Must start with / to be a valid relative path
        if url.startswith('/'):
            return True
        return False
    
    # For absolute URLs, only allow http/https
    if parsed.scheme and parsed.scheme.lower() not in {'http', 'https'}:
        return False
    
    # For absolute URLs, check the host is in the allowlist
    if parsed.netloc:
        # Extract hostname (without port)
        hostname = parsed.hostname
        if hostname and hostname.lower() in allowed_hosts:
            return True
        return False
    
    # Default deny
    return False
