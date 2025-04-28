from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.style import Style
from rich.text import Text
from typing import Dict, Optional
import threading # Import threading

console = Console()


class AgentProgress:
    """Manages progress tracking for multiple agents."""

    def __init__(self):
        self.agent_status: Dict[str, Dict[str, str]] = {}
        self.status_cache: Dict[str, str] = {}
        self.table = Table(show_header=False, box=None, padding=(0, 1))
        # Live watches self.table and updates automatically
        self.live = Live(self.table, console=console, refresh_per_second=4, vertical_overflow="visible")
        self.started = False
        self._lock = threading.Lock() # Add a lock

    def _build_table(self):
        """Builds the table content based on the current agent status.
           This method MUST be called *within* the lock.
        """
        # Clear existing content
        self.table.columns.clear()
        # Instead of self.table.rows.clear(), we'll just rebuild rows
        # Create a temporary list for new rows
        new_rows = []

        # Sort agents
        def sort_key(item):
            agent_name = item[0]
            if "risk_management" in agent_name: return (2, agent_name)
            if "portfolio_management" in agent_name: return (3, agent_name)
            return (1, agent_name)

        # Create row data based on the current state held in self.agent_status
        for agent_name, info in sorted(self.agent_status.items(), key=sort_key):
            status = info.get("status", "")
            ticker = info.get("ticker")

            # Determine style and symbol
            if status.lower() == "done":
                style, symbol = Style(color="green", bold=True), "✓"
            elif status.lower() == "error":
                style, symbol = Style(color="red", bold=True), "✗"
            else:
                style, symbol = Style(color="yellow"), "⋯"

            agent_display = agent_name.replace("_agent", "").replace("_", " ").title()
            status_text = Text()
            status_text.append(f"{symbol} ", style=style)
            status_text.append(f"{agent_display:<20}", style=Style(bold=True))
            if ticker: status_text.append(f"[{ticker}] ", style=Style(color="cyan"))
            status_text.append(status, style=style)
            # Append the renderable for the row to our temporary list
            new_rows.append(status_text)

        # Safely update the table's definition and rows
        # Re-add the column definition (needed after clear)
        self.table.add_column(width=100)
        # Assign the new rows in one go
        self.table.rows = new_rows


    def start(self):
        """Start the progress display."""
        with self._lock: # Protect access during start
            if not self.started:
                self._build_table() # Initial table build within lock
                self.live.start(refresh=True) # Start Live updates
                self.started = True

    def stop(self):
        """Stop the progress display."""
        with self._lock: # Protect access during stop
            if self.started:
                self._build_table() # Final table build before stopping
                # It's better to let Live handle the final render on stop
                self.live.stop()
                self.started = False


    def update_status(self, agent_name: str, ticker: Optional[str] = None, status: str = ""):
        """Update the status of an agent's internal state (thread-safe)."""
        cache_key = f"{agent_name}:{ticker}"

        with self._lock: # Acquire lock before checking cache and updating state/table
            # Check cache inside lock
            if cache_key in self.status_cache and self.status_cache[cache_key] == status:
                return

            self.status_cache[cache_key] = status

            if agent_name not in self.agent_status:
                self.agent_status[agent_name] = {"status": "", "ticker": None}

            # Update internal state
            if ticker: self.agent_status[agent_name]["ticker"] = ticker
            if status: self.agent_status[agent_name]["status"] = status

            # Rebuild the table content *now* that the state is updated
            # Live will pick up the changes on its next refresh cycle
            if self.started:
                 self._build_table()

# Global instance
progress = AgentProgress()
