-- Supabase SQL schema for storing trading data

-- Trading Cycles table - Main cycle metadata
CREATE TABLE trading_cycles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tickers JSONB NOT NULL DEFAULT '[]'::jsonb,
    portfolio_value_before DECIMAL(15, 2),
    portfolio_value_after DECIMAL(15, 2),
    cash_before DECIMAL(15, 2),
    cash_after DECIMAL(15, 2),
    cycle_data JSONB NOT NULL DEFAULT '{}'::jsonb
);

-- Portfolio Strategy table - Stock decisions
CREATE TABLE portfolio_strategy (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    cycle_id UUID REFERENCES trading_cycles(id),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ticker TEXT NOT NULL,
    action TEXT NOT NULL,  -- buy, sell, short, cover, hold
    confidence DECIMAL(5, 2) NOT NULL DEFAULT 0,
    reasoning TEXT,
    raw_data JSONB NOT NULL DEFAULT '{}'::jsonb
);

-- Options Analysis table - Options decisions
CREATE TABLE options_analysis (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    cycle_id UUID REFERENCES trading_cycles(id),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    underlying_ticker TEXT NOT NULL,
    action TEXT NOT NULL,  -- buy, sell, none
    option_ticker TEXT,
    details JSONB NOT NULL DEFAULT '{}'::jsonb,  -- option_type, strike_price, expiration_date
    strategy TEXT,  -- long_call, long_put, etc.
    confidence DECIMAL(5, 2) NOT NULL DEFAULT 0,
    limit_price DECIMAL(15, 2),
    reasoning TEXT,
    raw_data JSONB NOT NULL DEFAULT '{}'::jsonb
);

-- Execution Summary table - Stock executions
CREATE TABLE execution_summary (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    cycle_id UUID REFERENCES trading_cycles(id),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ticker TEXT NOT NULL,
    status TEXT NOT NULL,  -- executed, skipped, error
    action TEXT,  -- buy, sell, short, cover
    quantity INTEGER,
    order_status TEXT,
    order_id TEXT,
    message TEXT,
    raw_data JSONB NOT NULL DEFAULT '{}'::jsonb
);

-- Options Execution Summary table - Options executions
CREATE TABLE options_execution_summary (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    cycle_id UUID REFERENCES trading_cycles(id),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    option_ticker TEXT NOT NULL,
    status TEXT NOT NULL,  -- executed, skipped, error
    action TEXT,  -- buy, sell
    quantity INTEGER,
    type TEXT,  -- market, limit
    order_status TEXT,
    order_id TEXT,
    message TEXT,
    raw_data JSONB NOT NULL DEFAULT '{}'::jsonb
);

-- Position Management table - Stop loss and profit taking adjustments
CREATE TABLE position_management (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ticker TEXT NOT NULL,
    action_type TEXT NOT NULL,  -- adjust_stop, scale_in, scale_out, exit, hold
    status TEXT NOT NULL,  -- success, skipped, error
    quantity INTEGER,
    order_id TEXT,
    message TEXT,
    details JSONB NOT NULL DEFAULT '{}'::jsonb,  -- target price, etc.
    raw_data JSONB NOT NULL DEFAULT '{}'::jsonb
);

-- Create indexes for better query performance
CREATE INDEX idx_trading_cycles_timestamp ON trading_cycles(timestamp);
CREATE INDEX idx_portfolio_strategy_ticker ON portfolio_strategy(ticker);
CREATE INDEX idx_portfolio_strategy_cycle_id ON portfolio_strategy(cycle_id);
CREATE INDEX idx_options_analysis_underlying ON options_analysis(underlying_ticker);
CREATE INDEX idx_options_analysis_cycle_id ON options_analysis(cycle_id);
CREATE INDEX idx_execution_summary_ticker ON execution_summary(ticker);
CREATE INDEX idx_execution_summary_cycle_id ON execution_summary(cycle_id);
CREATE INDEX idx_options_execution_ticker ON options_execution_summary(option_ticker);
CREATE INDEX idx_options_execution_cycle_id ON options_execution_summary(cycle_id);
CREATE INDEX idx_position_management_ticker ON position_management(ticker); 