import { NextResponse } from 'next/server';
import Alpaca from '@alpacahq/alpaca-trade-api';

// Define an interface for the Alpaca position object for type safety
interface AlpacaPosition {
  symbol: string;
  qty: string;
  avg_entry_price: string;
  current_price: string;
  market_value: string;
  unrealized_pl: string;
  unrealized_plpc: string; // Alpaca API returns this as a string representing a decimal, e.g., "0.1" for 10%
}

export async function GET(_request: Request) { // Prefix unused 'request' with an underscore
  console.log('API route /api/portfolio called to fetch real Alpaca data');

  const alpaca = new Alpaca({
    keyId: process.env.ALPACA_API_KEY_ID,
    secretKey: process.env.ALPACA_API_SECRET_KEY,
    paper: true, // Assuming you are using a paper trading account for the demo
    baseUrl: process.env.ALPACA_PAPER_URL, // Ensure this is set in .env.local for paper
  });

  try {
    // Fetch account information
    const account = await alpaca.getAccount();

    // Fetch current positions
    const positionsRaw = await alpaca.getPositions();

    // Format positions data (similar to your mock structure)
    const positions = positionsRaw.map((pos: AlpacaPosition) => ({ // Use the specific AlpacaPosition type
      ticker: pos.symbol,
      quantity: parseFloat(pos.qty),
      entry_price: parseFloat(pos.avg_entry_price),
      current_price: parseFloat(pos.current_price),
      market_value: parseFloat(pos.market_value),
      unrealized_pl: parseFloat(pos.unrealized_pl),
      unrealized_pl_pct: parseFloat(pos.unrealized_plpc) * 100, // Alpaca gives plpc as a decimal (e.g., 0.1 for 10%)
    }));

    const portfolioData = {
      portfolio_value: parseFloat(account.portfolio_value),
      cash_balance: parseFloat(account.cash),
      equity: parseFloat(account.equity),
      buying_power: parseFloat(account.buying_power),
      positions: positions,
      timestamp: new Date().toISOString(),
    };

    return NextResponse.json(portfolioData);

  } catch (e: unknown) { // Type error as unknown for better safety
    let errorMessage = 'Failed to fetch portfolio data from Alpaca';
    let errorStatus = 500;
    let errorDetails = '';

    if (e instanceof Error) {
      errorMessage = e.message;
      errorDetails = e.toString();
      // Check if the error object has a statusCode property (common in HTTP clients)
      if ('statusCode' in e && typeof (e as { statusCode?: unknown }).statusCode === 'number') {
        errorStatus = (e as { statusCode: number }).statusCode;
      }
    } else if (typeof e === 'object' && e !== null) {
      // Attempt to extract message and statusCode if it's a non-Error object
      errorMessage = (e as { message?: string }).message || errorMessage;
      errorStatus = (e as { statusCode?: number }).statusCode || errorStatus;
      errorDetails = String(e);
    } else {
      // Fallback for other types of errors (e.g., string)
      errorDetails = String(e);
    }
    console.error('Error fetching portfolio data from Alpaca:', errorMessage, errorDetails);


    return NextResponse.json(
      { error: errorMessage, details: errorDetails },
      { status: errorStatus }
    );
  }
} 