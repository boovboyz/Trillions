import { NextResponse } from 'next/server';
import Alpaca from '@alpacahq/alpaca-trade-api';

export async function GET(request: Request) {
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
    const positions = positionsRaw.map((pos: any) => ({
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

  } catch (error: any) {
    console.error('Error fetching portfolio data from Alpaca:', error.message || error);
    // Check for specific Alpaca error structure if available
    const errorMessage = error.message || 'Failed to fetch portfolio data from Alpaca';
    const errorStatus = error.statusCode || 500;

    return NextResponse.json(
      { error: errorMessage, details: error.toString() },
      { status: errorStatus }
    );
  }
} 