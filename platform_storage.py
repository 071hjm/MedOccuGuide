from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


def _now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class StorageConfig:
    base_dir: Path = Path(r"D:\codex")
    db_path: Path = Path(r"D:\codex\platform_data.db")
    default_username: str = "admin"
    default_password: str = "admin123"
    default_display_name: str = "系统管理员"
    rule_suggestion_version: str = "expert_rules_v1"


class PlatformStorage:
    def __init__(self, config: StorageConfig | None = None):
        self.config = config or StorageConfig()
        self.config.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.config.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_column(self, conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
        columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sample_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    input_mode TEXT NOT NULL,
                    sample_id TEXT,
                    final_label TEXT,
                    final_source TEXT,
                    diagnosis TEXT,
                    advice_mode TEXT,
                    created_at TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    result_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS expert_case_library (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    final_label TEXT,
                    diagnosis TEXT,
                    note TEXT,
                    created_at TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    result_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS operation_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    action TEXT NOT NULL,
                    detail TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )

            self._ensure_column(conn, "expert_case_library", "rule_hit_tag", "rule_hit_tag TEXT")
            self._ensure_column(conn, "expert_case_library", "rule_version", "rule_version TEXT")

            existing = conn.execute(
                "SELECT username FROM users WHERE username = ?",
                (self.config.default_username,),
            ).fetchone()
            if existing is None:
                conn.execute(
                    "INSERT INTO users (username, password, display_name, created_at) VALUES (?, ?, ?, ?)",
                    (
                        self.config.default_username,
                        self.config.default_password,
                        self.config.default_display_name,
                        _now_text(),
                    ),
                )
            conn.commit()

    def authenticate(self, username: str, password: str) -> tuple[bool, str]:
        username = (username or "").strip()
        password = (password or "").strip()
        if not username or not password:
            return False, "请输入用户名和密码。"
        with self._connect() as conn:
            row = conn.execute(
                "SELECT username, password, display_name FROM users WHERE username = ?",
                (username,),
            ).fetchone()
        if row is None:
            return False, "用户名不存在。"
        if row["password"] != password:
            return False, "密码不正确。"
        return True, str(row["display_name"])

    @staticmethod
    def _compact_json(payload: Any) -> str:
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _extract_rule_hit_tag(result: dict[str, Any]) -> str:
        final_result = result.get("final_result") or {}
        if final_result.get("source") == "expert_system":
            rule_id = str(final_result.get("rule_id") or "").strip()
            return f"命中 {rule_id}" if rule_id else "命中"
        return "未命中"

    def save_analysis_record(
        self,
        username: str,
        input_mode: str,
        sample_id: str | None,
        result: dict[str, Any],
    ) -> int:
        final_result = result.get("final_result") or {}
        advice = result.get("advice") or {}
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO sample_records (
                    username, input_mode, sample_id, final_label, final_source,
                    diagnosis, advice_mode, created_at, record_json, result_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    username,
                    input_mode,
                    sample_id or "",
                    str(final_result.get("label") or ""),
                    str(final_result.get("source") or ""),
                    str(final_result.get("diagnosis") or ""),
                    str(advice.get("mode") or ""),
                    _now_text(),
                    self._compact_json(result.get("record") or {}),
                    self._compact_json(result),
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def save_expert_case(self, username: str, result: dict[str, Any], note: str = "") -> int:
        final_result = result.get("final_result") or {}
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO expert_case_library (
                    username, final_label, diagnosis, note, rule_hit_tag, rule_version,
                    created_at, record_json, result_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    username,
                    str(final_result.get("label") or ""),
                    str(final_result.get("diagnosis") or ""),
                    note.strip(),
                    self._extract_rule_hit_tag(result),
                    self.config.rule_suggestion_version,
                    _now_text(),
                    self._compact_json(result.get("record") or {}),
                    self._compact_json(result),
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def log_operation(self, username: str, action: str, detail: str = "") -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO operation_logs (
                    username, action, detail, created_at
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    (username or "").strip() or "anonymous",
                    (action or "").strip(),
                    (detail or "").strip(),
                    _now_text(),
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def list_sample_records(
        self,
        *,
        keyword: str = "",
        final_label: str = "",
        final_source: str = "",
        limit: int = 100,
    ) -> list[list[str]]:
        conditions: list[str] = []
        params: list[Any] = []
        keyword = keyword.strip()
        final_label = final_label.strip()
        final_source = final_source.strip()
        if keyword:
            conditions.append(
                "(username LIKE ? OR sample_id LIKE ? OR final_label LIKE ? OR diagnosis LIKE ? OR advice_mode LIKE ?)"
            )
            keyword_like = f"%{keyword}%"
            params.extend([keyword_like] * 5)
        if final_label:
            conditions.append("final_label = ?")
            params.append(final_label)
        if final_source:
            conditions.append("final_source = ?")
            params.append(final_source)
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"""
            SELECT id, created_at, username, input_mode, sample_id, final_label, final_source, advice_mode
            FROM sample_records
            {where_clause}
            ORDER BY id DESC
            LIMIT ?
        """
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [
            [
                str(row["id"]),
                str(row["created_at"]),
                str(row["username"]),
                str(row["input_mode"]),
                str(row["sample_id"]),
                str(row["final_label"]),
                str(row["final_source"]),
                str(row["advice_mode"]),
            ]
            for row in rows
        ]

    def list_expert_cases(
        self,
        *,
        keyword: str = "",
        final_label: str = "",
        rule_hit_tag: str = "",
        rule_version: str = "",
        limit: int = 100,
    ) -> list[list[str]]:
        conditions: list[str] = []
        params: list[Any] = []
        keyword = keyword.strip()
        final_label = final_label.strip()
        rule_hit_tag = rule_hit_tag.strip()
        rule_version = rule_version.strip()
        if keyword:
            conditions.append("(username LIKE ? OR final_label LIKE ? OR diagnosis LIKE ? OR note LIKE ?)")
            keyword_like = f"%{keyword}%"
            params.extend([keyword_like] * 4)
        if final_label:
            conditions.append("final_label = ?")
            params.append(final_label)
        if rule_hit_tag:
            conditions.append("rule_hit_tag LIKE ?")
            params.append(f"%{rule_hit_tag}%")
        if rule_version:
            conditions.append("rule_version LIKE ?")
            params.append(f"%{rule_version}%")
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"""
            SELECT id, created_at, username, final_label, diagnosis, rule_hit_tag, rule_version, note
            FROM expert_case_library
            {where_clause}
            ORDER BY id DESC
            LIMIT ?
        """
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [
            [
                str(row["id"]),
                str(row["created_at"]),
                str(row["username"]),
                str(row["final_label"]),
                str(row["diagnosis"]),
                str(row["rule_hit_tag"] or ""),
                str(row["rule_version"] or ""),
                str(row["note"]),
            ]
            for row in rows
        ]

    def list_operation_logs(
        self,
        *,
        keyword: str = "",
        action: str = "",
        username: str = "",
        limit: int = 200,
    ) -> list[list[str]]:
        conditions: list[str] = []
        params: list[Any] = []
        keyword = keyword.strip()
        action = action.strip()
        username = username.strip()
        if keyword:
            conditions.append("(username LIKE ? OR action LIKE ? OR detail LIKE ?)")
            keyword_like = f"%{keyword}%"
            params.extend([keyword_like] * 3)
        if action:
            conditions.append("action LIKE ?")
            params.append(f"%{action}%")
        if username:
            conditions.append("username LIKE ?")
            params.append(f"%{username}%")
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"""
            SELECT id, created_at, username, action, detail
            FROM operation_logs
            {where_clause}
            ORDER BY id DESC
            LIMIT ?
        """
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [
            [
                str(row["id"]),
                str(row["created_at"]),
                str(row["username"]),
                str(row["action"]),
                str(row["detail"]),
            ]
            for row in rows
        ]

    def get_sample_record_by_id(self, record_id: int | str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, username, input_mode, sample_id, final_label, created_at, record_json, result_json FROM sample_records WHERE id = ?",
                (int(record_id),),
            ).fetchone()
        if row is None:
            return None
        return {
            "id": int(row["id"]),
            "username": str(row["username"]),
            "input_mode": str(row["input_mode"]),
            "sample_id": str(row["sample_id"] or ""),
            "final_label": str(row["final_label"] or ""),
            "created_at": str(row["created_at"]),
            "record": json.loads(row["record_json"]),
            "result": json.loads(row["result_json"]),
        }

    def get_expert_case_by_id(self, case_id: int | str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, username, final_label, diagnosis, note, rule_hit_tag, rule_version, created_at, record_json, result_json
                FROM expert_case_library
                WHERE id = ?
                """,
                (int(case_id),),
            ).fetchone()
        if row is None:
            return None
        return {
            "id": int(row["id"]),
            "username": str(row["username"]),
            "final_label": str(row["final_label"] or ""),
            "diagnosis": str(row["diagnosis"] or ""),
            "note": str(row["note"] or ""),
            "rule_hit_tag": str(row["rule_hit_tag"] or ""),
            "rule_version": str(row["rule_version"] or ""),
            "created_at": str(row["created_at"]),
            "record": json.loads(row["record_json"]),
            "result": json.loads(row["result_json"]),
        }
