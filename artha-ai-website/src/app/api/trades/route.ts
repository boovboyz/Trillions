import { NextResponse } from 'next/server';
import Alpaca from '@alpacahq/alpaca-trade-api';

// Define an interface for the Alpaca order object for type safety
interface AlpacaOrder {
  id: string;
  symbol: string;
  side: string; // e.g., 'buy', 'sell'
  qty: string; // Quantity as a string, parsed to float later
  filled_avg_price: string; // Filled average price as a string, parsed to float later
  filled_at: string | null; // Timestamp as string, can be null
  submitted_at: string; // Timestamp as string
  status: string; // e.g., 'filled', 'open'
  // Add other properties if accessed from the order object
}

export async function GET(_request: Request) {
  console.log('API route /api/trades called to fetch recent trades');

  // Mock data - in a real implementation, these would come from Alpaca
  /* const mockTrades = [
    {
      id: '1',
      ticker: 'AAPL',
      type: 'buy',
      quantity: 10,
      price: 175.23,
      timestamp: new Date(Date.now() - 3600000).toISOString(), // 1 hour ago
      status: 'filled'
    },
    {
      id: '2',
      ticker: 'MSFT',
      type: 'sell',
      quantity: 5,
      price: 326.78,
      timestamp: new Date(Date.now() - 7200000).toISOString(), // 2 hours ago
      status: 'filled'
    },
    {
      id: '3',
      ticker: 'NVDA',
      type: 'buy',
      quantity: 8,
      price: 420.15,
      timestamp: new Date(Date.now() - 10800000).toISOString(), // 3 hours ago
      status: 'filled'
    },
    {
      id: '4',
      ticker: 'AMZN',
      type: 'buy',
      quantity: 12,
      price: 142.56,
      timestamp: new Date(Date.now() - 14400000).toISOString(), // 4 hours ago
      status: 'filled'
    },
    {
      id: '5',
      ticker: 'GOOGL',
      type: 'buy',
      quantity: 6,
      price: 162.34,
      timestamp: new Date(Date.now() - 18000000).toISOString(), // 5 hours ago
      status: 'filled'
    }
  ]; */

  // In a production environment, you would use the Alpaca API to fetch real trades:
  const alpaca = new Alpaca({
    keyId: process.env.ALPACA_API_KEY_ID,
    secretKey: process.env.ALPACA_API_SECRET_KEY,
    paper: true, // Assuming you are using a paper trading account for the demo
    baseUrl: process.env.ALPACA_PAPER_URL,
  });

  try {
    // Get most recent orders
    const orders = await alpaca.getOrders({
      status: 'filled', // 'open', 'closed', 'all'
      limit: 10, 
      direction: 'desc', // 'asc' or 'desc'
      nested: true, // if true, embeds an order's legs (if any) in the order record
      after: undefined, // Date | string
      until: undefined, // Date | string
      symbols: undefined // string | string[]
    });

    const trades = orders.map((order: AlpacaOrder) => ({
      id: order.id,
      ticker: order.symbol,
      type: order.side,
      quantity: parseFloat(order.qty),
      price: parseFloat(order.filled_avg_price),
      timestamp: order.filled_at || order.submitted_at, 
      status: order.status
    }));

    return NextResponse.json(trades);
  } catch (e: unknown) {
    let errorMessage = 'Failed to fetch trades data from Alpaca';
    let errorStatus = 500;
    let errorDetails = '';

    if (e instanceof Error) {
      errorMessage = e.message;
      errorDetails = e.toString();
      // Check if the error object has a statusCode property
      if ('statusCode' in e && typeof (e as { statusCode?: unknown }).statusCode === 'number') {
        errorStatus = (e as { statusCode: number }).statusCode;
      }
    } else if (typeof e === 'object' && e !== null) {
      // Attempt to extract message and statusCode if it's a non-Error object
      errorMessage = (e as { message?: string }).message || errorMessage;
      errorStatus = (e as { statusCode?: number }).statusCode || errorStatus;
      errorDetails = String(e);
    } else {
      // Fallback for other types of errors
      errorDetails = String(e);
    }
    console.error('Error fetching trades from Alpaca:', errorMessage, errorDetails);

    return NextResponse.json(
      { error: errorMessage, details: errorDetails },
      { status: errorStatus }
    );
  }
  // return NextResponse.json(mockTrades); // Ensure mock data is not returned
} 