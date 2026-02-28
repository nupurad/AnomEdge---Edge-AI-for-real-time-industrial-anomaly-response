import sqlite3
import os

DB_PATH = "data/edge_sentinel.db"  # change name if you want

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def reset_db():
    # Deletes the database file so everything is recreated fresh
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

def init_db(reset: bool = False):
    os.makedirs("data", exist_ok=True)

    if reset:
        reset_db()

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS incidents (
        id TEXT PRIMARY KEY,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,

        camera_id TEXT,
        zone TEXT,
        machine_id TEXT,

        anomaly_type TEXT NOT NULL,      -- normal|oil_leak|smoke|conveyor_jam
        severity TEXT NOT NULL,          -- P0|P1|P2
        confidence REAL,                 -- 0..1

        summary TEXT NOT NULL,
        sop_refs_json TEXT,
        plan_json TEXT,

        image_path TEXT,
        model_name TEXT,
        connectivity TEXT,               -- offline|online

        status TEXT NOT NULL DEFAULT 'open',
        resolved_at INTEGER
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS audit_events (
        id TEXT PRIMARY KEY,
        incident_id TEXT NOT NULL,
        timestamp INTEGER NOT NULL,

        event_type TEXT NOT NULL,        -- frame_captured|gemma_analyzed|sop_retrieved|tool_executed|sync_*
        data_json TEXT,

        FOREIGN KEY (incident_id) REFERENCES incidents(id)
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS outbox (
        id TEXT PRIMARY KEY,
        incident_id TEXT NOT NULL,

        event_type TEXT NOT NULL,        -- incident_created|incident_resolved
        payload_json TEXT NOT NULL,

        status TEXT NOT NULL DEFAULT 'pending',
        attempts INTEGER NOT NULL DEFAULT 0,
        next_attempt_at INTEGER,
        last_error TEXT,

        created_at INTEGER NOT NULL,

        FOREIGN KEY (incident_id) REFERENCES incidents(id)
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS cache (
        key TEXT PRIMARY KEY,            -- e.g. sop::smoke
        value_json TEXT NOT NULL,
        kind TEXT NOT NULL,              -- sop|llm
        created_at INTEGER NOT NULL,
        ttl_seconds INTEGER,
        hits INTEGER NOT NULL DEFAULT 0
    );
    """)

    # Helpful indexes (optional but recommended)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_incidents_created_at ON incidents(created_at);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_incidents_anomaly_type ON incidents(anomaly_type);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_incidents_status ON incidents(status);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_incident ON audit_events(incident_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_outbox_status ON outbox(status);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_cache_kind ON cache(kind);")

    conn.commit()
    conn.close()