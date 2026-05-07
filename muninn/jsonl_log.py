"""Simple append-only JSONL logger keyed by date and session_id."""
from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


class JsonlLogger:
    def __init__(self, logs_dir: Path, session_id: str | None = None) -> None:
        self.logs_dir = logs_dir
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id or uuid.uuid4().hex[:12]
        # Pin date and time at session start so a single session always lands
        # in one file even if it crosses midnight.
        now = datetime.now()
        self.session_date = now.strftime("%Y-%m-%d")
        self.session_time = now.strftime("%H%M%S")
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        # Layout: logs/<YYYY-MM-DD>/<session_id>-<HHMMSS>.jsonl
        # Subdir-per-day keeps the parent listable even after thousands of
        # sessions. Time suffix means a `ls` of the day-dir is human-readable
        # without having to open files to find when each session ran.
        return (
            self.logs_dir
            / self.session_date
            / f"{self.session_id}-{self.session_time}.jsonl"
        )

    def log(self, record: dict[str, Any]) -> None:
        record = {"ts": time.time(), "session_id": self.session_id, **record}
        line = json.dumps(record, default=str, ensure_ascii=False)
        target = self.path
        with self._lock:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
