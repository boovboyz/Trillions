import time
import threading

class RateLimiter:
    """Simple token-bucket rate limiter."""
    def __init__(self, max_calls: int, period: float):
        self.max_calls = max_calls
        self.period = period
        self.lock = threading.Lock()
        self.calls: list[float] = []

    def acquire(self):
        with self.lock:
            now = time.monotonic()
            # Remove timestamps older than period
            self.calls = [t for t in self.calls if now - t < self.period]
            if len(self.calls) >= self.max_calls:
                sleep_for = self.period - (now - self.calls[0])
                time.sleep(sleep_for)
            self.calls.append(time.monotonic())