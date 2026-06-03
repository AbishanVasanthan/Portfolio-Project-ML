"""
db.py — Direct PostgreSQL client via psycopg.

Drop-in replacement for supabase-py: the same
  get_client().table("name").select().eq().execute()
API works unchanged across the rest of the codebase.
Requires DATABASE_URL in the environment (same connection
string the dashboard backend already uses).
"""
import logging
import os
import threading
from decimal import Decimal
from typing import Any

import psycopg
from psycopg.rows import dict_row

logger = logging.getLogger(__name__)


def _coerce(row: dict) -> dict:
    """Convert Decimal → float so pandas arithmetic works without surprises."""
    return {k: float(v) if isinstance(v, Decimal) else v for k, v in row.items()}


def _db_url() -> str:
    url = os.environ["DATABASE_URL"]
    return url.replace("postgresql+psycopg://", "postgresql://").replace(
        "postgres+psycopg://", "postgresql://"
    )


# ── Thread-local persistent connection ────────────────────────
# Reuses one connection per thread — avoids exhausting Windows
# socket quota when hundreds of queries fire in rapid succession.

_local = threading.local()


def _get_conn() -> psycopg.Connection:
    conn = getattr(_local, "conn", None)
    if conn is not None and not conn.closed:
        try:
            conn.execute("SELECT 1")
            return conn
        except Exception:
            try:
                conn.close()
            except Exception:
                pass
    _local.conn = psycopg.connect(_db_url(), row_factory=dict_row)
    logger.debug("[DB] New connection opened")
    return _local.conn


# ── Result ─────────────────────────────────────────────────────

class Result:
    __slots__ = ("data", "count")

    def __init__(self, data: Any, count: int | None = None):
        self.data = data
        self.count = count


# ── Query builder ──────────────────────────────────────────────

class QueryBuilder:
    def __init__(self, table: str):
        self._table = table
        self._cols = "*"
        self._count_exact = False
        self._wheres: list[tuple] = []
        self._orders: list[tuple[str, bool]] = []
        self._limit_n: int | None = None
        self._offset_n: int | None = None
        self._action = "select"
        self._payload: list[dict] | dict | None = None
        self._on_conflict = ""
        self._single = False
        self._maybe = False

    # ── Filters / ordering ─────────────────────────────────────

    def select(self, cols: str = "*", count: str | None = None):
        self._cols = cols
        self._count_exact = count == "exact"
        return self

    def eq(self, col: str, val: Any):
        self._wheres.append((col, "=", val))
        return self

    def neq(self, col: str, val: Any):
        self._wheres.append((col, "!=", val))
        return self

    def gt(self, col: str, val: Any):
        self._wheres.append((col, ">", val))
        return self

    def is_(self, col: str, val: str):
        op = "IS NULL" if val == "null" else "IS NOT NULL"
        self._wheres.append((col, op, None))
        return self

    def order(self, col: str, desc: bool = False):
        self._orders.append((col, desc))
        return self

    def limit(self, n: int):
        self._limit_n = n
        return self

    def range(self, start: int, end: int):
        self._offset_n = start
        self._limit_n = end - start + 1
        return self

    def single(self):
        self._single = True
        return self

    def maybe_single(self):
        self._maybe = True
        return self

    # ── Mutations ──────────────────────────────────────────────

    def insert(self, data: dict | list[dict]):
        self._action = "insert"
        self._payload = data if isinstance(data, list) else [data]
        return self

    def upsert(self, data: dict | list[dict], on_conflict: str = ""):
        self._action = "upsert"
        self._payload = data if isinstance(data, list) else [data]
        self._on_conflict = on_conflict
        return self

    def update(self, data: dict):
        self._action = "update"
        self._payload = data
        return self

    def delete(self):
        self._action = "delete"
        return self

    # ── Execution ──────────────────────────────────────────────

    def execute(self) -> Result:
        conn = _get_conn()
        if self._action == "select":
            return self._do_select(conn)
        if self._action == "insert":
            result = self._do_insert(conn)
            conn.commit()
            return result
        if self._action == "upsert":
            result = self._do_upsert(conn)
            conn.commit()
            return result
        if self._action == "update":
            result = self._do_update(conn)
            conn.commit()
            return result
        if self._action == "delete":
            result = self._do_delete(conn)
            conn.commit()
            return result
        raise RuntimeError(f"Unknown DB action: {self._action}")

    # ── Private SQL builders ───────────────────────────────────

    def _where(self) -> tuple[str, list]:
        if not self._wheres:
            return "", []
        parts, params = [], []
        for col, op, val in self._wheres:
            if op in ("IS NULL", "IS NOT NULL"):
                parts.append(f"{col} {op}")
            else:
                parts.append(f"{col} {op} %s")
                params.append(val)
        return "WHERE " + " AND ".join(parts), params

    def _order(self) -> str:
        if not self._orders:
            return ""
        parts = [f"{col} {'DESC' if d else 'ASC'}" for col, d in self._orders]
        return "ORDER BY " + ", ".join(parts)

    def _do_select(self, conn: psycopg.Connection) -> Result:
        sel = "*" if self._cols == "*" else ", ".join(
            c.strip() for c in self._cols.split(",")
        )
        where, params = self._where()
        parts = [f"SELECT {sel} FROM {self._table}", where, self._order()]
        if self._limit_n:
            parts.append(f"LIMIT {self._limit_n}")
        if self._offset_n:
            parts.append(f"OFFSET {self._offset_n}")
        sql = " ".join(p for p in parts if p)

        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        count = None
        if self._count_exact:
            csql = " ".join(p for p in [
                f"SELECT COUNT(*) AS n FROM {self._table}", where
            ] if p)
            with conn.cursor() as cur:
                cur.execute(csql, params)
                row = cur.fetchone()
                count = int(row["n"]) if row else 0

        rows = [_coerce(r) for r in rows]
        if self._single:
            return Result(data=rows[0] if rows else {})
        if self._maybe:
            return Result(data=rows[0] if rows else None)
        return Result(data=rows, count=count)

    def _do_insert(self, conn: psycopg.Connection) -> Result:
        rows: list[dict] = self._payload  # type: ignore
        if not rows:
            return Result(data=[])
        cols = list(rows[0].keys())
        col_str = ", ".join(cols)
        row_ph = "(" + ", ".join(["%s"] * len(cols)) + ")"
        values_str = ", ".join([row_ph] * len(rows))
        vals = [r[c] for r in rows for c in cols]
        sql = f"INSERT INTO {self._table} ({col_str}) VALUES {values_str} RETURNING *"
        with conn.cursor() as cur:
            cur.execute(sql, vals)
            return Result(data=[_coerce(r) for r in cur.fetchall()])

    def _do_upsert(self, conn: psycopg.Connection) -> Result:  # noqa
        rows: list[dict] = self._payload  # type: ignore
        if not rows:
            return Result(data=[])
        cols = list(rows[0].keys())
        conflict_cols = [c.strip() for c in self._on_conflict.split(",") if c.strip()]
        update_cols = [c for c in cols if c not in conflict_cols]
        col_str = ", ".join(cols)
        row_ph = "(" + ", ".join(["%s"] * len(cols)) + ")"
        values_str = ", ".join([row_ph] * len(rows))
        vals = [r[c] for r in rows for c in cols]

        if conflict_cols and update_cols:
            sql = (
                f"INSERT INTO {self._table} ({col_str}) VALUES {values_str} "
                f"ON CONFLICT ({', '.join(conflict_cols)}) DO UPDATE SET "
                + ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)
                + " RETURNING *"
            )
        elif conflict_cols:
            sql = (
                f"INSERT INTO {self._table} ({col_str}) VALUES {values_str} "
                f"ON CONFLICT ({', '.join(conflict_cols)}) DO NOTHING RETURNING *"
            )
        else:
            sql = f"INSERT INTO {self._table} ({col_str}) VALUES {values_str} RETURNING *"

        with conn.cursor() as cur:
            cur.execute(sql, vals)
            return Result(data=[_coerce(r) for r in cur.fetchall()])

    def _do_update(self, conn: psycopg.Connection) -> Result:
        data: dict = self._payload  # type: ignore
        if not data:
            return Result(data=[])
        where, where_params = self._where()
        set_str = ", ".join(f"{c} = %s" for c in data)
        sql = f"UPDATE {self._table} SET {set_str} {where} RETURNING *"
        with conn.cursor() as cur:
            cur.execute(sql, list(data.values()) + where_params)
            return Result(data=[_coerce(r) for r in cur.fetchall()])

    def _do_delete(self, conn: psycopg.Connection) -> Result:
        where, params = self._where()
        sql = f"DELETE FROM {self._table} {where} RETURNING *"
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return Result(data=cur.fetchall())


# ── Public API ─────────────────────────────────────────────────

class _DBClient:
    """Supabase-py drop-in. Usage: get_client().table("x").select().eq().execute()"""
    def table(self, name: str) -> QueryBuilder:
        return QueryBuilder(name)


_client: _DBClient | None = None


def get_client() -> _DBClient:
    global _client
    if _client is None:
        _client = _DBClient()
        logger.info("[DB] PostgreSQL client initialised (direct psycopg)")
    return _client


def check_schema() -> bool:
    try:
        get_client().table("tc_depots").select("depot_id").limit(1).execute()
        return True
    except Exception:
        return False
