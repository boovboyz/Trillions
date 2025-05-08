# Supabase Integration for AI Trading Platform

This integration allows the AI trading platform to store trading data in Supabase. The following data will be stored:

1. Portfolio Strategy - The buy/sell/short/cover decisions for each stock
2. Options Analysis - The options strategies and contract selections
3. Execution Summary - The results of stock order executions
4. Position Management - The stop loss adjustments and position scaling actions

## Implementation Notes

This integration uses direct HTTP requests to the Supabase REST API instead of the official `supabase-py` client. This approach was chosen to avoid dependency conflicts between the Supabase client and Alpaca Trade API, which both require different versions of the `websockets` package.

## Setup Instructions

### 1. Create a Supabase Account

1. Go to [Supabase](https://supabase.com/) and sign up for an account
2. Create a new project
3. Note your project URL and API key (found in Project Settings > API)

### 2. Set Up the Database Schema

1. In the Supabase dashboard, go to the SQL Editor
2. Copy the contents of `supabase_schema.sql` and run the SQL to create the tables
3. Verify that the following tables were created:
   - `trading_cycles`
   - `portfolio_strategy`
   - `options_analysis`
   - `execution_summary`
   - `options_execution_summary`
   - `position_management`

### 3. Configure Environment Variables

Add your Supabase credentials to your `.env` file:

```
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your-supabase-api-key
```

### 4. Install Requirements

Make sure the `requests` library is installed (it probably already is):

```
pip install requests
```

Or add it to your `requirements.txt` or `pyproject.toml` file.

### 5. Enable Supabase Integration

Run the trading manager with the `--enable-supabase` flag:

```bash
python src/trading_manager.py --tickers AAPL,MSFT --enable-options --trading-frequency hourly --enable-supabase
```

## Viewing the Data

You can view the collected data in the Supabase dashboard:

1. Go to your Supabase project
2. Select "Table Editor" from the sidebar
3. Browse the tables listed above to see the stored trading data

## Querying the Data

Here are some sample SQL queries to analyze your trading data:

### Get All Trading Cycles

```sql
SELECT 
  id, 
  timestamp, 
  tickers,
  portfolio_value_before,
  portfolio_value_after,
  (portfolio_value_after - portfolio_value_before) as profit_loss,
  ((portfolio_value_after / portfolio_value_before) - 1) * 100 as roi_percent
FROM 
  trading_cycles
ORDER BY 
  timestamp DESC;
```

### Get All Portfolio Strategy Decisions

```sql
SELECT 
  ps.ticker, 
  ps.action, 
  ps.confidence, 
  ps.timestamp,
  tc.portfolio_value_before
FROM 
  portfolio_strategy ps
JOIN 
  trading_cycles tc ON ps.cycle_id = tc.id
ORDER BY 
  ps.timestamp DESC;
```

### Get Options Analysis by Underlying Stock

```sql
SELECT 
  underlying_ticker,
  action,
  strategy,
  confidence,
  details->>'option_type' as option_type,
  details->>'strike_price' as strike_price,
  details->>'expiration_date' as expiration_date,
  timestamp
FROM 
  options_analysis
WHERE 
  underlying_ticker = 'AAPL'
ORDER BY 
  timestamp DESC;
```

### Get Execution Results

```sql
SELECT 
  ticker,
  action,
  status,
  quantity,
  order_status,
  timestamp
FROM 
  execution_summary
ORDER BY 
  timestamp DESC;
```

### Get Position Management Actions

```sql
SELECT 
  ticker,
  action_type,
  status,
  message,
  timestamp
FROM 
  position_management
ORDER BY 
  timestamp DESC;
```

## Using the Data Analyzer Tool

The project includes a data analyzer tool to query and visualize the data from Supabase. To use it:

```bash
# List the most recent trading cycles
python src/tools/supabase_analyzer.py cycles

# View portfolio strategy decisions for a specific ticker
python src/tools/supabase_analyzer.py strategy --ticker AAPL

# Generate a performance report
python src/tools/supabase_analyzer.py report

# Plot portfolio performance
python src/tools/supabase_analyzer.py plot-portfolio
```

## Advanced Usage

### Automatic Position Management Storage

Position management actions are automatically stored in Supabase when the `--enable-supabase` flag is used. This includes:

- Stop loss adjustments
- Take profit adjustments
- Position scaling (increasing/decreasing position size) 
- Position exits

### Raw Data Storage

All tables include a `raw_data` JSONB column that stores the complete data structure for each record. This allows for maximum flexibility in data analysis, as you can access any field from the original data structures.

To query specific fields from the raw data:

```sql
SELECT 
  ticker,
  raw_data->>'reasoning' as reasoning
FROM 
  portfolio_strategy
WHERE 
  action = 'buy';
``` 