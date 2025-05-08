import { NextResponse } from 'next/server';
import { createClient } from '@supabase/supabase-js';

// Initialize Supabase client
// Note: For server-side routes, using the service_role key is common for full access.
// Ensure these are set in your .env.local file.
const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY!; // Use service role key for backend access
const supabase = createClient(supabaseUrl, supabaseKey);

interface StrategyItemBase {
  id: string;
  timestamp: string;
  action: string;
  confidence: number | null;
  reasoning: string | null;
  // analysts_involved will be tricky with this new approach, placeholder for now
}

interface StockStrategyItem extends StrategyItemBase {
  item_type: 'Stock';
  ticker: string;
}

interface OptionStrategyItem extends StrategyItemBase {
  item_type: 'Option';
  underlying_ticker: string;
  option_ticker: string | null;
  strategy: string | null; 
  details: any; // JSONB from schema for option details
}

// Combined type for Supabase RPC or view if we used one
type CombinedStrategyItem = StockStrategyItem | OptionStrategyItem;

// This interface should match the structure returned by your SQL function
interface RpcStrategyItem {
  id: string; 
  item_type: 'Stock' | 'Option';
  ticker: string; // For stocks or underlying_ticker for options
  option_ticker: string | null;
  action: string;
  timestamp: string; // Supabase returns TIMESTAMPTZ as string by default
  confidence: number | null;
  reasoning: string | null;
  strategy: string | null; // For options
  details: any | null; // For options (JSONB)
}

interface FormattedReasoning {
  id: string;
  item_type: 'Stock' | 'Option';
  ticker: string;
  option_ticker?: string | null; 
  action: string;
  timestamp: string;
  reasoning_summary: string;
  analysts_involved: string[];
  confidence?: number | null;
  strategy?: string | null;
  option_details?: any | null;
}

export async function GET(request: Request) {
  console.log('API route /api/reasoning called to fetch most recent analysis per ticker.');

  try {
    // Using Supabase RPC to execute a more complex query for DISTINCT ON
    // This is often easier than trying to replicate DISTINCT ON with PostgREST js library features directly.
    // You would need to create this SQL function in your Supabase SQL editor.
    /*
    Example SQL Function (put this in Supabase SQL Editor under a schema, e.g., public):

    CREATE OR REPLACE FUNCTION get_latest_strategy_per_ticker()
    RETURNS TABLE (
        id UUID,
        item_type TEXT,
        ticker TEXT, -- underlying for options, direct for stocks
        option_ticker TEXT, -- null for stocks
        action TEXT,
        "timestamp" TIMESTAMPTZ, -- ensure correct quoting if needed for column name
        confidence DECIMAL,
        reasoning TEXT,
        strategy TEXT, -- null for stocks
        details JSONB -- null for stocks
    )
    AS $$
    BEGIN
        RETURN QUERY
        SELECT s.id, 'Stock' as item_type, s.ticker, NULL as option_ticker, s.action, s."timestamp", s.confidence, s.reasoning, NULL as strategy, NULL as details
        FROM ( 
            SELECT *, ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY "timestamp" DESC) as rn
            FROM portfolio_strategy
        ) s
        WHERE s.rn = 1

        UNION ALL

        SELECT o.id, 'Option' as item_type, o.underlying_ticker as ticker, o.option_ticker, o.action, o."timestamp", o.confidence, o.reasoning, o.strategy, o.details
        FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY underlying_ticker ORDER BY "timestamp" DESC) as rn
            FROM options_analysis
        ) o
        WHERE o.rn = 1;
    END;
    $$ LANGUAGE plpgsql;

    */

    const { data: rpcData, error: rpcError } = await supabase
      .rpc('get_latest_strategy_per_ticker');

    if (rpcError) {
      console.error('Supabase RPC error for get_latest_strategy_per_ticker:', rpcError.message);
      throw rpcError;
    }

    // Cast the data from RPC to our defined type. Supabase rpc returns `any[] | null` by default.
    const combinedData = rpcData as RpcStrategyItem[] | null;

    if (!combinedData) {
      return NextResponse.json({ reasoning: [] });
    }
    
    const sortedData = combinedData.sort((a: RpcStrategyItem, b: RpcStrategyItem) => 
      new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
    );

    const formattedReasoning: FormattedReasoning[] = sortedData.map((item: RpcStrategyItem) => ({
      id: item.id,
      item_type: item.item_type,
      ticker: item.ticker, 
      option_ticker: item.item_type === 'Option' ? item.option_ticker : null,
      action: item.action,
      timestamp: item.timestamp,
      reasoning_summary: item.reasoning || 'No reasoning provided.',
      analysts_involved: [], // Placeholder - to be addressed if data is available
      confidence: item.confidence,
      strategy: item.item_type === 'Option' ? item.strategy : null,
      option_details: item.item_type === 'Option' ? item.details : null,
    }));

    return NextResponse.json(formattedReasoning.slice(0, 15));

  } catch (error: any) {
    console.error('Error in /api/reasoning:', error.message || error);
    return NextResponse.json(
      { error: 'Failed to fetch reasoning data from Supabase', details: error.toString() },
      { status: 500 }
    );
  }
} 