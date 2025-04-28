from pydantic import BaseModel

class Price(BaseModel):
    open: float
    close: float
    high: float
    low: float
    volume: int
    time: str

class PriceResponse(BaseModel):
    ticker: str
    prices: list[Price]

class FinancialMetrics(BaseModel):
    ticker: str
    report_period: str
    period: str
    currency: str
    market_cap: float | None = None
    revenue: float | None = None
    net_income: float | None = None
    # Allow additional fields
    model_config = {"extra": "allow"}

class FinancialMetricsResponse(BaseModel):
    financial_metrics: list[FinancialMetrics]

class LineItem(BaseModel):
    ticker: str
    report_period: str
    period: str
    currency: str
    # Explicitly include requested line items for direct attribute access
    revenue: float | None = None
    earnings_per_share: float | None = None
    free_cash_flow: float | None = None
    net_income: float | None = None
    depreciation_and_amortization: float | None = None
    capital_expenditure: float | None = None
    working_capital: float | None = None
    dividends_and_other_cash_distributions: float | None = None
    book_value_per_share: float | None = None
    debt_to_equity: float | None = None
    total_liabilities: float | None = None
    total_assets: float | None = None
    ebit: float | None = None
    ebitda: float | None = None
    operating_margin: float | None = None
    research_and_development: float | None = None
    gross_margin: float | None = None
    shareholders_equity: float | None = None
    total_debt: float | None = None
    cash_and_equivalents: float | None = None
    current_assets: float | None = None
    current_liabilities: float | None = None
    outstanding_shares: float | None = None
    # Allow other dynamic fields
    model_config = {"extra": "allow"}

class LineItemResponse(BaseModel):
    search_results: list[LineItem]

class InsiderTrade(BaseModel):
    ticker: str
    transaction_date: str | None = None
    filing_date: str
    insider_name: str | None = None
    transaction_type: str | None = None
    shares: float | None = None
    price: float | None = None
    # Allow additional fields
    model_config = {"extra": "allow"}

class InsiderTradeResponse(BaseModel):
    insider_trades: list[InsiderTrade]

class CompanyNews(BaseModel):
    ticker: str
    title: str
    author: str | None = None
    source: str | None = None
    date: str
    url: str | None = None
    text: str | None = None
    # Allow additional fields
    model_config = {"extra": "allow"}

class CompanyNewsResponse(BaseModel):
    news: list[CompanyNews]

class CompanyFacts(BaseModel):
    ticker: str
    company_name: str | None = None
    cik: str | None = None
    market_cap: float | None = None
    employees: int | None = None
    website_url: str | None = None
    # Allow additional fields
    model_config = {"extra": "allow"}

class CompanyFactsResponse(BaseModel):
    company_facts: CompanyFacts


class Position(BaseModel):
    cash: float = 0.0
    shares: int = 0
    ticker: str


class Portfolio(BaseModel):
    positions: dict[str, Position]  # ticker -> Position mapping
    total_cash: float = 0.0


class AnalystSignal(BaseModel):
    signal: str | None = None
    confidence: float | None = None
    reasoning: dict | str | None = None
    max_position_size: float | None = None  # For risk management signals


class TickerAnalysis(BaseModel):
    ticker: str
    analyst_signals: dict[str, AnalystSignal]  # agent_name -> signal mapping


class AgentStateData(BaseModel):
    tickers: list[str]
    portfolio: Portfolio
    start_date: str
    end_date: str
    ticker_analyses: dict[str, TickerAnalysis]  # ticker -> analysis mapping


class AgentStateMetadata(BaseModel):
    show_reasoning: bool = False
    model_config = {"extra": "allow"}
