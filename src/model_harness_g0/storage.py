from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class Ledger:
    def __init__(self, root: Path):
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root = root
        self.db = sqlite3.connect(root / "ledger.sqlite", check_same_thread=False)
        os.chmod(root / "ledger.sqlite", 0o600)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("""CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY, trial_id TEXT UNIQUE NOT NULL, key_hash TEXT UNIQUE NOT NULL,
            state TEXT NOT NULL, created REAL NOT NULL, touched REAL NOT NULL,
            reward REAL, reward_completion TEXT, reason TEXT)""")
        # In-memory engine caches cannot be reconstructed after process death.
        self.db.execute(
            "UPDATE sessions SET state='QUARANTINED', reason='server_restart' WHERE state='ACTIVE'"
        )
        self.db.commit()

    def add(self, sid: str, trial: str, key: str):
        now = time.time()
        with self.db:
            self.db.execute(
                "INSERT INTO sessions VALUES (?,?,?,?,?,?,?,?,?)",
                (sid, trial, digest(key), "ACTIVE", now, now, None, None, None),
            )

    def by_key(self, key: str):
        return self.db.execute("SELECT * FROM sessions WHERE key_hash=?", (digest(key),)).fetchone()

    def get(self, sid: str):
        return self.db.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()

    def state(self, sid: str, state: str, reason: str | None = None):
        with self.db:
            self.db.execute(
                "UPDATE sessions SET state=?, reason=?, touched=? WHERE id=?",
                (state, reason, time.time(), sid),
            )

    def touch(self, sid: str):
        with self.db:
            self.db.execute("UPDATE sessions SET touched=? WHERE id=?", (time.time(), sid))

    def reward(self, sid: str, value: float, completion: str):
        row = self.get(sid)
        if row["reward"] is not None:
            if row["reward"] != value or row["reward_completion"] != completion:
                raise ValueError("conflicting final reward")
            return False
        with self.db:
            self.db.execute(
                "UPDATE sessions SET reward=?, reward_completion=? WHERE id=?",
                (value, completion, sid),
            )
        return True

    def rows(self):
        return [dict(r) for r in self.db.execute("SELECT * FROM sessions ORDER BY created")]

    def close(self):
        self.db.close()
