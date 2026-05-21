import os
import logging
from psycopg2 import pool

logger = logging.getLogger(__name__)

_pool: pool.ThreadedConnectionPool | None = None


def get_pool() -> pool.ThreadedConnectionPool:
    global _pool
    if _pool is None:
        url = os.environ["DATABASE_URL"]
        _pool = pool.ThreadedConnectionPool(minconn=1, maxconn=10, dsn=url)
        logger.info("[DB] Connection pool initialised")
    return _pool


def get_conn():
    return get_pool().getconn()


def release_conn(conn):
    get_pool().putconn(conn)


def execute_sql_file(path: str) -> None:
    conn = get_conn()
    try:
        with open(path, "r") as f:
            sql = f.read()
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
        logger.info("[DB] Executed SQL file: %s", path)
    except Exception:
        conn.rollback()
        raise
    finally:
        release_conn(conn)
