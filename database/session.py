"""
SQLite session management for GateHub.

Vehicles enter from any gate; the session stays open until they exit
from any gate.  No CNIC is held — the guard scans it at entry, the
number is stored, and the card is returned immediately.
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DB_PATH     = Path('gatehub.db')
SCHEMA_PATH = Path('database/schema.sql')

OVERSTAY_MINUTES = 90


# ── helpers ──────────────────────────────────────────────────────────

def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


# ── setup ─────────────────────────────────────────────────────────────

def init_db() -> None:
    """Create tables if they don't exist yet."""
    conn = _connect()
    conn.executescript(SCHEMA_PATH.read_text())
    conn.commit()
    conn.close()


# ── entry / exit ──────────────────────────────────────────────────────

def vehicle_entry(plate: str, entry_gate: str, cnic: Optional[str] = None) -> int:
    """
    Log a vehicle arriving at a gate.
    Returns the new session id.
    """
    conn   = _connect()
    cursor = conn.execute(
        'INSERT INTO vehicles (plate, cnic, entry_gate, entry_time, status) '
        'VALUES (?, ?, ?, ?, ?)',
        (plate.upper(), cnic, entry_gate, _now(), 'active'),
    )
    session_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return session_id


def vehicle_exit(plate: str, exit_gate: str) -> Optional[dict]:
    """
    Close the most recent active session for a plate.
    Returns the completed session row, or None if no active session found.
    """
    conn = _connect()
    row  = conn.execute(
        "SELECT * FROM vehicles "
        "WHERE plate = ? AND status = 'active' "
        "ORDER BY entry_time DESC LIMIT 1",
        (plate.upper(),),
    ).fetchone()

    if row is None:
        conn.close()
        return None

    conn.execute(
        "UPDATE vehicles SET exit_gate = ?, exit_time = ?, status = 'exited' "
        "WHERE id = ?",
        (exit_gate, _now(), row['id']),
    )
    conn.commit()
    conn.close()
    return dict(row)


# ── queries ───────────────────────────────────────────────────────────

def get_active_sessions() -> list[dict]:
    """Return all active vehicle sessions with elapsed_min computed."""
    conn = _connect()
    rows = conn.execute(
        "SELECT *, "
        "  ROUND((strftime('%s','now') - strftime('%s', entry_time)) / 60.0, 1) "
        "  AS elapsed_min "
        "FROM vehicles "
        "WHERE status = 'active' "
        "ORDER BY entry_time ASC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── blacklist ─────────────────────────────────────────────────────────

def is_blacklisted(plate: str) -> bool:
    conn  = _connect()
    found = conn.execute(
        'SELECT 1 FROM blacklist WHERE plate = ?', (plate.upper(),)
    ).fetchone()
    conn.close()
    return found is not None


def add_to_blacklist(plate: str, reason: str = '') -> None:
    conn = _connect()
    conn.execute(
        'INSERT OR REPLACE INTO blacklist (plate, reason, added_at) VALUES (?, ?, ?)',
        (plate.upper(), reason, _now()),
    )
    conn.commit()
    conn.close()


def remove_from_blacklist(plate: str) -> None:
    conn = _connect()
    conn.execute('DELETE FROM blacklist WHERE plate = ?', (plate.upper(),))
    conn.commit()
    conn.close()


def get_blacklist() -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        'SELECT * FROM blacklist ORDER BY added_at DESC'
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
