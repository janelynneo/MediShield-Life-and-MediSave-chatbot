"""
Authentication utilities — password verification and role-based access.
Uses bcrypt directly for Python 3.14 compatibility.
"""

import json
import os
from pathlib import Path

import bcrypt
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent.parent
load_dotenv(BASE_DIR / ".env")


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

    Returns True if credentials are valid, False otherwise.
    """
    users = _load_users()
    entry = users.get(username)
    if not entry:
        return False

    stored_hash = entry.get("password", "")
    if not stored_hash:
        return False

    try:
        return bcrypt.checkpw(password.encode(), stored_hash.encode())
    except Exception:
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
