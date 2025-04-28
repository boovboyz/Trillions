import datetime
import os
import logging
import time
import random
from functools import wraps

import requests
import pandas as pd
import pandas_market_calendars as mcal

from src.data.cache import get_cache
from src.utils.rate_limiter import RateLimiter
from src.data.models import (
    Price, PriceResponse,
    FinancialMetrics, FinancialMetricsResponse,
    LineItem, LineItemResponse,
    InsiderTrade, InsiderTradeResponse,
    CompanyNews, CompanyNewsResponse,
    CompanyFactsResponse
)

# Global cache and rate limiter
_cache = get_cache()
_rate_limiter = RateLimiter(max_calls=5, period=1.0)  # 5 requests/sec


def retry_with_backoff(retries=5, backoff_in_seconds=1, jitter=0.1):
    """Decorator for retrying API calls with exponential backoff and jitter, handling HTTP errors and 429s."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < retries:
                try:
                    return func(*args, **kwargs)
                except requests.HTTPError as e:
                    status = e.response.status_code if e.response is not None else None
                    # Handle rate limits specifically
                    if status == 429:
                        retry_after = e.response.headers.get('Retry-After')
                        sleep_time = float(retry_after) if retry_after else backoff_in_seconds * (2 ** attempts)
                    else:
                        sleep_time = backoff_in_seconds * (2 ** attempts)
                    sleep_time += random.uniform(0, jitter)
                    logging.warning(f"API error {status}, retrying {func.__name__} in {sleep_time:.2f}s...")
                    time.sleep(sleep_time)
                    attempts += 1
            logging.error(f"{func.__name__} failed after {retries} attempts.")
            raise
        return wrapper
    return decorator


@retry_with_backoff()
def make_request(method: str, url: str, **kwargs) -> requests.Response:
    """HTTP request wrapper with rate limiting and error handling."""
    _rate_limiter.acquire()
    resp = requests.request(method, url, timeout=10, **kwargs)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        logging.error(f"HTTP {resp.status_code} on {url}: {resp.text}")
        raise
    return resp


# --- Prices ---
def get_prices(ticker: str, start_date: str, end_date: str) -> list[Price]:
    """Fetch daily price data, caching missing intervals persistently."""
    # Determine trading days via NYSE calendar
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date=start_date, end_date=end_date)
    expected = {d.strftime('%Y-%m-%d') for d in schedule.index}

    cached = _cache.get_prices(ticker) or []
    cached_dates = {item['time'] for item in cached}
    missing = sorted(expected - cached_dates)

    if missing:
        logging.info(f"Fetching missing {len(missing)} days for {ticker}.")
        api_key = os.getenv('FINANCIAL_DATASETS_API_KEY')
        headers = {'X-API-KEY': api_key} if api_key else {}
        url = (
            f"https://api.financialdatasets.ai/prices/"
            f"?ticker={ticker}&interval=day&interval_multiplier=1"
            f"&start_date={start_date}&end_date={end_date}"
        )
        resp = make_request('GET', url, headers=headers)
        new_prices = PriceResponse(**resp.json()).prices
        raw = [p.model_dump() for p in new_prices]
        _cache.set_prices(ticker, raw)
        cached.extend(raw)

    result = [Price(**p) for p in cached if start_date <= p['time'] <= end_date]
    return sorted(result, key=lambda x: x.time)


def prices_to_df(prices: list[Price]) -> pd.DataFrame:
    """Convert Price list to a timezone-aware DataFrame with data validation."""
    if not prices:
        return pd.DataFrame() # Return empty DF if no prices
        
    df = pd.DataFrame([p.model_dump() for p in prices])
    df['Date'] = pd.to_datetime(df['time'], utc=True)
    df.set_index('Date', inplace=True)

    # Check for and remove duplicate index entries before proceeding
    if df.index.has_duplicates:
        duplicates_count = df.index.duplicated().sum()
        logging.warning(f"Found {duplicates_count} duplicate date entries in price data. Keeping first occurrence.")
        df = df[~df.index.duplicated(keep='first')]
        
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    nan_frac = df[['open','high','low','close','volume']].isna().mean().max()
    if nan_frac > 0.05:
        raise ValueError(f"Too many NaNs in price data: {nan_frac:.2%}")
    df = df.tz_convert('America/New_York')
    biz = mcal.get_calendar('NYSE').schedule(
        start_date=df.index.min().date(), end_date=df.index.max().date()
    ).index
    biz_index = pd.DatetimeIndex(biz).tz_localize('America/New_York')
    df = df.reindex(biz_index)
    df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].ffill()
    return df


def get_price_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Helper to fetch prices and return a DataFrame."""
    return prices_to_df(get_prices(ticker, start_date, end_date))


# --- Financial Metrics ---
def get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = 'ttm',
    limit: int = 10,
) -> list[FinancialMetrics]:
    """Fetch financial metrics snapshot, caching persistently."""
    cached = _cache.get_financial_metrics(ticker) or []
    filtered = [FinancialMetrics(**m) for m in cached if m['report_period'] <= end_date]
    if filtered:
        logging.info(f"Cache hit for metrics: {ticker} (<= {end_date})")
        return sorted(filtered, key=lambda x: x.report_period, reverse=True)[:limit]

    api_key = os.getenv('FINANCIAL_DATASETS_API_KEY')
    headers = {'X-API-KEY': api_key} if api_key else {}
    url = (
        f"https://api.financialdatasets.ai/financial-metrics"
        f"?ticker={ticker}&period={period}&limit={limit}&report_period_lte={end_date}"
    )
    resp = make_request('GET', url, headers=headers)
    metrics = FinancialMetricsResponse(**resp.json()).financial_metrics
    raw = [m.model_dump() for m in metrics]
    _cache.set_financial_metrics(ticker, raw)
    return metrics


# --- Line Items Search ---
def search_line_items(
    ticker: str,
    line_items: list[str],
    end_date: str,
    period: str = 'ttm',
    limit: int = 10,
) -> list[LineItem]:
    """Fetch specific financial line items via single request."""
    cached = _cache.get_line_items(ticker) or []
    if cached:
        logging.info(f"Cache hit for line items: {ticker}")
        return [LineItem(**li) for li in cached][:limit]

    api_key = os.getenv('FINANCIAL_DATASETS_API_KEY')
    headers = {'X-API-KEY': api_key} if api_key else {}
    url = 'https://api.financialdatasets.ai/financials/search/line-items'
    body = {
        'tickers': [ticker],
        'line_items': line_items,
        'period': period,
        'end_date': end_date,
        'limit': limit,
    }
    resp = make_request('POST', url, json=body, headers=headers)
    items = LineItemResponse(**resp.json()).search_results
    raw = [li.model_dump() for li in items]
    _cache.set_line_items(ticker, raw)
    return items


# --- Insider Trades ---
def get_insider_trades(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
) -> list[InsiderTrade]:
    """Fetch insider trades for a ticker, filtering by filing_date."""
    cached = _cache.get_insider_trades(ticker) or []
    filtered = [InsiderTrade(**t) for t in cached
                if (start_date is None or t.get('filing_date') >= start_date)
                and t.get('filing_date') <= end_date]
    if filtered:
        logging.info(f"Cache hit for insider trades: {ticker}")
        return sorted(filtered, key=lambda x: x.filing_date, reverse=True)[:limit]

    api_key = os.getenv('FINANCIAL_DATASETS_API_KEY')
    headers = {'X-API-KEY': api_key} if api_key else {}
    url = (
        f"https://api.financialdatasets.ai/insider-trades"
        f"?ticker={ticker}&limit={limit}"
    )
    if start_date:
        url += f"&filing_date_gte={start_date}"
    url += f"&filing_date_lte={end_date}"
    resp = make_request('GET', url, headers=headers)
    trades = InsiderTradeResponse(**resp.json()).insider_trades
    raw = [t.model_dump() for t in trades]
    _cache.set_insider_trades(ticker, raw)
    return trades


# --- Company News ---
def get_company_news(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
) -> list[CompanyNews]:
    """Fetch company news with pagination."""
    cached = _cache.get_company_news(ticker) or []
    filtered = [CompanyNews(**n) for n in cached
                if (start_date is None or n['date'] >= start_date)
                and n['date'] <= end_date]
    if filtered:
        return sorted(filtered, key=lambda x: x.date, reverse=True)[:limit]

    api_key = os.getenv('FINANCIAL_DATASETS_API_KEY')
    headers = {'X-API-KEY': api_key} if api_key else {}
    all_news = []
    page = 1
    while len(all_news) < limit:
        url = (
            f"https://api.financialdatasets.ai/news"
            f"?ticker={ticker}&limit={limit}&page={page}&end_date={end_date}"
        )
        try:
            resp = make_request('GET', url, headers=headers)
            page_news = CompanyNewsResponse(**resp.json()).news
        except requests.HTTPError as e:
            # If persistent 404 after retries, assume endpoint issue and return empty list
            if e.response is not None and e.response.status_code == 404:
                logging.warning(f"News endpoint returned 404 for {ticker} after retries. Returning empty list.")
                _cache.set_company_news(ticker, []) # Cache empty result
                return []
            else:
                logging.error(f"Unhandled HTTP error fetching news for {ticker}: {e}")
                raise # Re-raise other HTTP errors
        except Exception as e:
            logging.error(f"Error processing news response for {ticker}: {e}")
            raise # Re-raise other unexpected errors

        if not page_news:
            break
        all_news.extend(page_news)
        page += 1
    raw = [n.model_dump() for n in all_news]
    _cache.set_company_news(ticker, raw)
    return all_news[:limit]


# --- Market Cap ---
def get_market_cap(
    ticker: str,
    end_date: str,
) -> float | None:
    """Fetch market cap, preferring CompanyFacts on today."""
    today = datetime.datetime.utcnow().strftime('%Y-%m-%d')
    if end_date == today:
        api_key = os.getenv('FINANCIAL_DATASETS_API_KEY')
        headers = {'X-API-KEY': api_key} if api_key else {}
        url = f"https://api.financialdatasets.ai/company/facts?ticker={ticker}"
        resp = make_request('GET', url, headers=headers)
        facts = CompanyFactsResponse(**resp.json()).company_facts
        if facts.market_cap is not None:
            return facts.market_cap
    metrics = get_financial_metrics(ticker, end_date, limit=1)
    return metrics[0].market_cap if metrics else None


# --- Preloading ---
def preload_data_for_tickers(
    tickers: list[str],
    start_date: str,
    end_date: str,
    preload_config: dict | None = None,
    max_workers: int = 5,
) -> list[str]:
    """Concurrent preload into cache for multiple tickers."""
    if preload_config is None:
        preload_config = {}
    failed = []
    import concurrent.futures
    logging.info(f"Preloading {len(tickers)} tickers...")
    def _load(tk):
        try:
            if preload_config.get('prices', True): get_prices(tk, start_date, end_date)
            if preload_config.get('metrics', True): get_financial_metrics(tk, end_date,
                                           period=preload_config.get('metrics_period','ttm'),
                                           limit=preload_config.get('metrics_limit',10))
            if preload_config.get('line_items'): search_line_items(tk, preload_config['line_items'], end_date,
                                           period=preload_config.get('metrics_period','ttm'),
                                           limit=preload_config.get('metrics_limit',10))
            if preload_config.get('insider_trades', True): get_insider_trades(tk, end_date, start_date)
            if preload_config.get('company_news', True): get_company_news(tk, end_date, start_date)
            if preload_config.get('market_cap', False): get_market_cap(tk, end_date)
        except Exception as e:
            logging.error(f"Preload failed for {tk}: {e}")
            return tk
        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for res in executor.map(_load, tickers):
            if res: failed.append(res)
    return failed
