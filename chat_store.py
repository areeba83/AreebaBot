"""
chat_store.py
-------------
Lightweight persistence layer for AreebaBot's chat history.

Saves every chat session (a list of {role, content} messages) to a local
JSON file (chat_sessions.json) so previous conversations can be reopened
later, similar to Claude's "Recents" sidebar.

This is plain local file storage - simple, dependency-free, and good
enough for a single-user local/demo deployment.
"""

import json
import os
import uuid
from datetime import datetime

STORE_PATH = "chat_sessions.json"


def _load_all():
    """Returns the full sessions dict: {session_id: {title, messages, created_at}}"""
    if not os.path.exists(STORE_PATH):
        return {}
    try:
        with open(STORE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_all(sessions):
    with open(STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(sessions, f, ensure_ascii=False, indent=2)


def list_sessions():
    """Returns sessions sorted newest-first as a list of (session_id, title, created_at)."""
    sessions = _load_all()
    items = [
        (sid, data.get("title", "New chat"), data.get("created_at", ""))
        for sid, data in sessions.items()
    ]
    items.sort(key=lambda x: x[2], reverse=True)
    return items


def create_session():
    """Creates a new empty session and returns its id."""
    sessions = _load_all()
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "title": "New chat",
        "messages": [],
        "created_at": datetime.now().isoformat(),
    }
    _save_all(sessions)
    return session_id


def get_messages(session_id):
    sessions = _load_all()
    return sessions.get(session_id, {}).get("messages", [])


def save_messages(session_id, messages):
    """Saves messages for a session and auto-titles it from the first user message."""
    sessions = _load_all()
    if session_id not in sessions:
        sessions[session_id] = {
            "title": "New chat",
            "messages": [],
            "created_at": datetime.now().isoformat(),
        }

    sessions[session_id]["messages"] = messages

    # Auto-title the chat using the first user message (like Claude does)
    if sessions[session_id]["title"] == "New chat":
        for msg in messages:
            if msg["role"] == "user":
                title = msg["content"].strip()
                sessions[session_id]["title"] = (
                    title[:40] + "..." if len(title) > 40 else title
                )
                break

    _save_all(sessions)


def delete_session(session_id):
    sessions = _load_all()
    if session_id in sessions:
        del sessions[session_id]
        _save_all(sessions)