-- GateHub SQLite schema

CREATE TABLE IF NOT EXISTS vehicles (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    plate       TEXT    NOT NULL,
    cnic        TEXT,                           -- nullable; not all entries scan CNIC
    entry_gate  TEXT    NOT NULL,
    entry_time  TEXT    NOT NULL,               -- ISO-8601 datetime string
    exit_gate   TEXT,
    exit_time   TEXT,
    status      TEXT    NOT NULL DEFAULT 'active'  -- 'active' | 'exited'
);

-- Fast lookup: find the active session for a plate at exit
CREATE INDEX IF NOT EXISTS idx_plate_status
    ON vehicles (plate, status);

CREATE TABLE IF NOT EXISTS blacklist (
    plate       TEXT PRIMARY KEY,
    reason      TEXT,
    added_at    TEXT NOT NULL
);
