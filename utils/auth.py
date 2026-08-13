"""
Authentication utilities — password verification and role-based access.
Uses bcrypt directly for Python 3.14 compatibility.
"""

import json
import os
import time
from pathlib import Path

import bcrypt
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent.parent
load_dotenv(BASE_DIR / ".env")

# Rate limiting: track failed attempts per username/IP
# Structure: {username: {"count": N, "first_attempt": timestamp, "blocked_until": timestamp}}
_failed_attempts: dict = {}
MAX_ATTEMPTS = 5
LOCKOUT_SECONDS = 300  # 5 minutes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_users() -> dict:
    """Load users from st.secrets (cloud) or USERS_JSON env var (local)."""
    try:
        import streamlit as st
        raw = st.secrets.get("USERS_JSON", "")
    except ImportError:
        raw = os.environ.get("USERS_JSON", "{}")

    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------------------
# Auth functions
# ---------------------------------------------------------------------------

def verify_user(username: str, password: str) -> bool:
    """
    Verify username + password against stored bcrypt hashes.
    Enforces rate limiting: after MAX_ATTEMPTS failed attempts within LOCKOUT_SECONDS,
    the account is locked for LOCKOUT_SECONDS.

    Returns True if credentials are valid, False otherwise.
    """
    now = time.time()

    # Check if account is currently locked
    if username in _failed_attempts:
        attempt_info = _failed_attempts[username]
        if now < attempt_info.get("blocked_until", 0):
            return False  # still locked out

    users = _load_users()
    entry = users.get(username)
    if not entry:
        return False

    stored_hash = entry.get("password", "")
    if not stored_hash:
        return False

    try:
        valid = bcrypt.checkpw(password.encode(), stored_hash.encode())
    except Exception:
        return False

    if valid:
        # Clear failed attempts on success
        _failed_attempts.pop(username, None)
        return True
    else:
        # Record failed attempt
        if username not in _failed_attempts:
            _failed_attempts[username] = {"count": 0, "first_attempt": now, "blocked_until": 0}
        info = _failed_attempts[username]
        info["count"] += 1
        # Reset if previous lockout expired
        if now > info.get("blocked_until", 0):
            info["count"] = 1
            info["first_attempt"] = now
        # Lock out if exceeded max attempts
        if info["count"] > MAX_ATTEMPTS:
            info["blocked_until"] = now + LOCKOUT_SECONDS
        _failed_attempts[username] = info
        return False


def get_role(username: str) -> str | None:
    """Return the user's role ('admin' or 'user'), or None if not found."""
    users = _load_users()
    entry = users.get(username)
    if not entry:
        return None
    return entry.get("role", "user")


def hash_password(password: str) -> str:
    """Generate a bcrypt hash for a plain-text password. Use for initial setup."""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
