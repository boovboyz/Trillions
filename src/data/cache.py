import sqlite3
import json
import threading
import time
from pathlib import Path
import logging

# Configure module logger
logger = logging.getLogger(__name__)

class Cache:
    """Disk-backed cache using SQLite for financial data."""
    def __init__(self, db_path: str = None):
        # Default to a cache.db next to this file
        if db_path is None:
            db_path = Path(__file__).parent.parent.parent / 'cache.db'
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.lock = threading.Lock()
        self._create_tables()

    def _create_tables(self):
        with self.lock, self.conn:
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS prices (
                ticker TEXT,
                time TEXT,
                data TEXT,
                PRIMARY KEY (ticker, time)
            )""")
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS financial_metrics (
                ticker TEXT,
                report_period TEXT,
                data TEXT,
                PRIMARY KEY (ticker, report_period)
            )""")
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS line_items (
                ticker TEXT,
                report_period TEXT,
                data TEXT,
                PRIMARY KEY (ticker, report_period)
            )""")
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS insider_trades (
                ticker TEXT,
                transaction_date TEXT,
                data TEXT,
                PRIMARY KEY (ticker, transaction_date)
            )""")
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS company_news (
                ticker TEXT,
                date TEXT,
                data TEXT,
                PRIMARY KEY (ticker, date)
            )""")

    def clear_all_tables(self):
        """Clear all data from cache tables without deleting the file."""
        with self.lock, self.conn:
            tables = ["prices", "financial_metrics", "line_items", "insider_trades", "company_news"]
            for table in tables:
                self.conn.execute(f"DELETE FROM {table}")
            self.conn.commit()
            logger.info("Cache tables cleared successfully")
            return True

    def close(self):
        """Close the database connection properly."""
        if self.conn:
            try:
                self.conn.close()
                logger.info("Cache connection closed successfully")
                return True
            except Exception as e:
                logger.error(f"Error closing cache connection: {e}")
                return False
        return True  # Already closed

    def get_prices(self, ticker: str):
        with self.lock:
            cur = self.conn.execute(
                "SELECT data FROM prices WHERE ticker = ? ORDER BY time",
                (ticker,)
            )
            rows = cur.fetchall()
        return [json.loads(r[0]) for r in rows] if rows else None

    def set_prices(self, ticker: str, data: list[dict]):
        with self.lock, self.conn:
            for item in data:
                self.conn.execute(
                    "INSERT OR IGNORE INTO prices (ticker, time, data) VALUES (?, ?, ?)",
                    (ticker, item['time'], json.dumps(item))
                )
    
    def get_financial_metrics(self, ticker: str):
        with self.lock:
            cur = self.conn.execute(
                "SELECT data FROM financial_metrics WHERE ticker = ? ORDER BY report_period",
                (ticker,)
            )
            rows = cur.fetchall()
        return [json.loads(r[0]) for r in rows] if rows else None

    def set_financial_metrics(self, ticker: str, data: list[dict]):
        with self.lock, self.conn:
            for item in data:
                self.conn.execute(
                    "INSERT OR IGNORE INTO financial_metrics (ticker, report_period, data) VALUES (?, ?, ?)",
                    (ticker, item['report_period'], json.dumps(item))
                )

    def get_line_items(self, ticker: str):
        with self.lock:
            cur = self.conn.execute(
                "SELECT data FROM line_items WHERE ticker = ? ORDER BY report_period",
                (ticker,)
            )
            rows = cur.fetchall()
        return [json.loads(r[0]) for r in rows] if rows else None

    def set_line_items(self, ticker: str, data: list[dict]):
        with self.lock, self.conn:
            for item in data:
                self.conn.execute(
                    "INSERT OR IGNORE INTO line_items (ticker, report_period, data) VALUES (?, ?, ?)",
                    (ticker, item['report_period'], json.dumps(item))
                )

    def get_insider_trades(self, ticker: str):
        with self.lock:
            cur = self.conn.execute(
                "SELECT data FROM insider_trades WHERE ticker = ? ORDER BY transaction_date",
                (ticker,)
            )
            rows = cur.fetchall()
        return [json.loads(r[0]) for r in rows] if rows else None

    def set_insider_trades(self, ticker: str, data: list[dict]):
        with self.lock, self.conn:
            for item in data:
                self.conn.execute(
                    "INSERT OR IGNORE INTO insider_trades (ticker, transaction_date, data) VALUES (?, ?, ?)",
                    (ticker, item['transaction_date'], json.dumps(item))
                )

    def get_company_news(self, ticker: str):
        with self.lock:
            cur = self.conn.execute(
                "SELECT data FROM company_news WHERE ticker = ? ORDER BY date",
                (ticker,)
            )
            rows = cur.fetchall()
        return [json.loads(r[0]) for r in rows] if rows else None

    def set_company_news(self, ticker: str, data: list[dict]):
        with self.lock, self.conn:
            for item in data:
                self.conn.execute(
                    "INSERT OR IGNORE INTO company_news (ticker, date, data) VALUES (?, ?, ?)",
                    (ticker, item['date'], json.dumps(item))
                )

# Global singleton - Initialize lazily
_cache: Cache | None = None

def get_cache() -> Cache:
    global _cache
    if _cache is None:
        _cache = Cache()
    return _cache

def close_cache():
    """Explicitly close the database connection and reset the singleton."""
    global _cache
    if _cache is not None:
        result = _cache.close()
        _cache = None
        return result
    return True

def clear_cache():
    """Clear all tables in the cache without deleting the file."""
    cache = get_cache()
    try:
        result = cache.clear_all_tables()
        return result
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return False