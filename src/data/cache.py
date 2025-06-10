import sqlite3
import json
import threading
import time
from pathlib import Path
import logging
from datetime import datetime, timedelta

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
            # Position management tables
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS position_tracking (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                position_type TEXT NOT NULL,  -- stock, option
                side TEXT NOT NULL,  -- long, short
                entry_price REAL NOT NULL,
                current_price REAL,
                quantity INTEGER NOT NULL,
                stop_loss REAL,
                take_profit REAL,
                trailing_stop_active BOOLEAN DEFAULT 0,
                trailing_stop_distance REAL,
                highest_price REAL,  -- For trailing stop tracking
                lowest_price REAL,   -- For short positions
                last_scale_price REAL,  -- Track last scale in/out price
                scale_count INTEGER DEFAULT 0,
                status TEXT DEFAULT 'active',  -- active, closed
                close_reason TEXT,
                close_price REAL,
                close_timestamp TEXT,
                pnl REAL,
                pnl_percent REAL,
                data TEXT  -- JSON for additional data
            )""")
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS position_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                position_id INTEGER,
                timestamp TEXT NOT NULL,
                action TEXT NOT NULL,  -- scale_in, scale_out, adjust_stop, adjust_target, close
                ticker TEXT NOT NULL,
                quantity INTEGER,
                price REAL,
                reason TEXT,
                confidence REAL,
                executed BOOLEAN DEFAULT 0,
                order_id TEXT,
                result TEXT,  -- success, failed, skipped
                data TEXT,  -- JSON for additional data
                FOREIGN KEY (position_id) REFERENCES position_tracking(id)
            )""")
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS profit_levels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                position_id INTEGER,
                level_number INTEGER NOT NULL,
                target_pnl_percent REAL NOT NULL,
                target_price REAL NOT NULL,
                quantity_percent REAL NOT NULL,  -- % of position to sell
                triggered BOOLEAN DEFAULT 0,
                trigger_timestamp TEXT,
                trigger_price REAL,
                data TEXT,  -- JSON for additional data
                FOREIGN KEY (position_id) REFERENCES position_tracking(id)
            )""")
            # Create indexes for better performance
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_position_ticker ON position_tracking(ticker)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_position_status ON position_tracking(status)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_position_timestamp ON position_tracking(timestamp)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_actions_ticker ON position_actions(ticker)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_actions_position ON position_actions(position_id)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_profit_position ON profit_levels(position_id)")

    def clear_all_tables(self):
        """Clear all data from cache tables without deleting the file."""
        with self.lock, self.conn:
            tables = ["prices", "financial_metrics", "line_items", "insider_trades", "company_news",
                     "position_tracking", "position_actions", "profit_levels"]
            for table in tables:
                self.conn.execute(f"DELETE FROM {table}")
            self.conn.commit()
            logger.info("Cache tables cleared successfully")
            return True
    
    def clear_market_data_tables(self):
        """Clear only market data tables, preserving position management data."""
        with self.lock, self.conn:
            market_data_tables = ["prices", "financial_metrics", "line_items", "insider_trades", "company_news"]
            for table in market_data_tables:
                self.conn.execute(f"DELETE FROM {table}")
            self.conn.commit()
            logger.info("Market data tables cleared successfully (position management data preserved)")
            return True
    
    def clear_old_market_data(self, hours_old: float = 1.167):
        """Clear market data older than specified hours."""
        from datetime import datetime, timedelta
        
        cutoff_time = (datetime.now() - timedelta(hours=hours_old)).isoformat()
        
        with self.lock, self.conn:
            market_data_tables = ["prices", "financial_metrics", "line_items", "insider_trades", "company_news"]
            cleared_count = 0
            
            for table in market_data_tables:
                try:
                    # Check if table has a time/timestamp column
                    cursor = self.conn.execute(f"PRAGMA table_info({table})")
                    columns = [row[1] for row in cursor.fetchall()]
                    
                    time_column = None
                    if 'time' in columns:
                        time_column = 'time'
                    elif 'timestamp' in columns:
                        time_column = 'timestamp'
                    elif 'date' in columns:
                        time_column = 'date'
                    elif 'transaction_date' in columns:
                        time_column = 'transaction_date'
                    
                    if time_column:
                        result = self.conn.execute(
                            f"DELETE FROM {table} WHERE {time_column} < ?",
                            (cutoff_time,)
                        )
                        cleared_count += result.rowcount
                        logger.info(f"Cleared {result.rowcount} old records from {table}")
                    else:
                        # If no time column, clear all data (it will be refreshed)
                        result = self.conn.execute(f"DELETE FROM {table}")
                        cleared_count += result.rowcount
                        logger.info(f"Cleared all {result.rowcount} records from {table} (no time column)")
                        
                except Exception as e:
                    logger.warning(f"Error clearing old data from {table}: {e}")
            
            self.conn.commit()
            logger.info(f"Cleared {cleared_count} old market data records (keeping fresh data)")
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
    
    # Position management methods
    def create_position_tracking(self, ticker: str, position_type: str, side: str, entry_price: float,
                               quantity: int, stop_loss: float = None, take_profit: float = None,
                               additional_data: dict = None) -> int:
        """Create a new position tracking record."""
        with self.lock, self.conn:
            cursor = self.conn.execute(
                """INSERT INTO position_tracking 
                (ticker, timestamp, position_type, side, entry_price, current_price, 
                 quantity, stop_loss, take_profit, highest_price, lowest_price, data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (ticker, datetime.now().isoformat(), position_type, side, entry_price, 
                 entry_price, quantity, stop_loss, take_profit, 
                 entry_price if side == 'long' else None,
                 entry_price if side == 'short' else None,
                 json.dumps(additional_data) if additional_data else None)
            )
            return cursor.lastrowid
    
    def get_active_position(self, ticker: str) -> dict:
        """Get active position for a ticker."""
        with self.lock:
            cursor = self.conn.execute(
                """SELECT * FROM position_tracking 
                WHERE ticker = ? AND status = 'active' 
                ORDER BY timestamp DESC LIMIT 1""",
                (ticker,)
            )
            row = cursor.fetchone()
            if row:
                columns = [desc[0] for desc in cursor.description]
                position = dict(zip(columns, row))
                if position.get('data'):
                    position['data'] = json.loads(position['data'])
                return position
            return None
    
    def update_position_price(self, position_id: int, current_price: float) -> None:
        """Update current price and high/low watermarks."""
        with self.lock:
            # Get current position
            cursor = self.conn.execute(
                "SELECT side, highest_price, lowest_price FROM position_tracking WHERE id = ?",
                (position_id,)
            )
            row = cursor.fetchone()
            if row:
                side, highest, lowest = row
                new_highest = max(highest or current_price, current_price) if side == 'long' else highest
                new_lowest = min(lowest or current_price, current_price) if side == 'short' else lowest
                
                with self.conn:
                    self.conn.execute(
                        """UPDATE position_tracking 
                        SET current_price = ?, highest_price = ?, lowest_price = ?
                        WHERE id = ?""",
                        (current_price, new_highest, new_lowest, position_id)
                    )
    
    def update_position_stops(self, position_id: int, stop_loss: float = None, take_profit: float = None,
                            trailing_stop_active: bool = None, trailing_stop_distance: float = None) -> None:
        """Update position stop loss and take profit levels."""
        with self.lock, self.conn:
            updates = []
            params = []
            
            if stop_loss is not None:
                updates.append("stop_loss = ?")
                params.append(stop_loss)
            if take_profit is not None:
                updates.append("take_profit = ?")
                params.append(take_profit)
            if trailing_stop_active is not None:
                updates.append("trailing_stop_active = ?")
                params.append(1 if trailing_stop_active else 0)
            if trailing_stop_distance is not None:
                updates.append("trailing_stop_distance = ?")
                params.append(trailing_stop_distance)
                
            if updates:
                params.append(position_id)
                self.conn.execute(
                    f"UPDATE position_tracking SET {', '.join(updates)} WHERE id = ?",
                    params
                )
    
    def record_position_action(self, position_id: int, action: str, ticker: str, 
                             quantity: int = None, price: float = None, reason: str = None,
                             confidence: float = None, order_id: str = None, 
                             result: str = 'pending', additional_data: dict = None) -> int:
        """Record a position management action."""
        with self.lock, self.conn:
            cursor = self.conn.execute(
                """INSERT INTO position_actions 
                (position_id, timestamp, action, ticker, quantity, price, reason, 
                 confidence, order_id, result, data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (position_id, datetime.now().isoformat(), action, ticker, quantity, 
                 price, reason, confidence, order_id, result,
                 json.dumps(additional_data) if additional_data else None)
            )
            return cursor.lastrowid
    
    def update_action_result(self, action_id: int, executed: bool, result: str, order_id: str = None) -> None:
        """Update the result of a position action."""
        with self.lock, self.conn:
            self.conn.execute(
                """UPDATE position_actions 
                SET executed = ?, result = ?, order_id = COALESCE(?, order_id)
                WHERE id = ?""",
                (1 if executed else 0, result, order_id, action_id)
            )
    
    def close_position(self, position_id: int, close_reason: str, close_price: float) -> None:
        """Close a position and calculate P&L."""
        with self.lock:
            cursor = self.conn.execute(
                """SELECT entry_price, quantity, side FROM position_tracking WHERE id = ?""",
                (position_id,)
            )
            row = cursor.fetchone()
            if row:
                entry_price, quantity, side = row
                
                # Calculate P&L
                if side == 'long':
                    pnl = (close_price - entry_price) * quantity
                    pnl_percent = ((close_price / entry_price) - 1) * 100
                else:  # short
                    pnl = (entry_price - close_price) * quantity
                    pnl_percent = ((entry_price / close_price) - 1) * 100
                
                with self.conn:
                    self.conn.execute(
                        """UPDATE position_tracking 
                        SET status = 'closed', close_reason = ?, close_price = ?, 
                            close_timestamp = ?, pnl = ?, pnl_percent = ?
                        WHERE id = ?""",
                        (close_reason, close_price, datetime.now().isoformat(), 
                         pnl, pnl_percent, position_id)
                    )
    
    def create_profit_levels(self, ticker: str, position_id: int, levels: list[dict]) -> None:
        """Create profit-taking levels for a position."""
        with self.lock, self.conn:
            for i, level in enumerate(levels):
                self.conn.execute(
                    """INSERT INTO profit_levels 
                    (ticker, position_id, level_number, target_pnl_percent, 
                     target_price, quantity_percent, data)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (ticker, position_id, i + 1, level['pnl_percent'], 
                     level['target_price'], level['quantity_percent'],
                     json.dumps(level.get('data')) if level.get('data') else None)
                )
    
    def get_profit_levels(self, position_id: int) -> list[dict]:
        """Get profit levels for a position."""
        with self.lock:
            cursor = self.conn.execute(
                """SELECT * FROM profit_levels 
                WHERE position_id = ? AND triggered = 0
                ORDER BY level_number""",
                (position_id,)
            )
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            levels = []
            for row in rows:
                level = dict(zip(columns, row))
                if level.get('data'):
                    level['data'] = json.loads(level['data'])
                levels.append(level)
            return levels
    
    def trigger_profit_level(self, level_id: int, trigger_price: float) -> None:
        """Mark a profit level as triggered."""
        with self.lock, self.conn:
            self.conn.execute(
                """UPDATE profit_levels 
                SET triggered = 1, trigger_timestamp = ?, trigger_price = ?
                WHERE id = ?""",
                (datetime.now().isoformat(), trigger_price, level_id)
            )
    
    def get_position_history(self, ticker: str = None, days: int = 30) -> list[dict]:
        """Get position history for analysis."""
        with self.lock:
            since = (datetime.now() - timedelta(days=days)).isoformat()
            if ticker:
                cursor = self.conn.execute(
                    """SELECT * FROM position_tracking 
                    WHERE ticker = ? AND timestamp >= ?
                    ORDER BY timestamp DESC""",
                    (ticker, since)
                )
            else:
                cursor = self.conn.execute(
                    """SELECT * FROM position_tracking 
                    WHERE timestamp >= ?
                    ORDER BY timestamp DESC""",
                    (since,)
                )
            
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            positions = []
            for row in rows:
                pos = dict(zip(columns, row))
                if pos.get('data'):
                    pos['data'] = json.loads(pos['data'])
                positions.append(pos)
            return positions

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
