'use client';

import { useState, useEffect } from 'react';
import { ArrowUpRight, ArrowDownRight, MinusCircle, Users, BarChart } from 'lucide-react'; // Import icons

interface ReasoningItem {
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
  option_details?: Record<string, unknown> | null;
}

interface ReasoningDisplayProps {
  preloadedData?: ReasoningItem[] | null;
}

// Helper to get an icon based on action
const getActionIcon = (action: string) => {
  const upperAction = action.toUpperCase();
  if (upperAction.includes('BUY') || upperAction.includes('COVER')) {
    return <ArrowUpRight className="w-4 h-4 inline-block mr-1 text-green-600" />;
  }
  if (upperAction.includes('SELL') || upperAction.includes('SHORT')) {
    return <ArrowDownRight className="w-4 h-4 inline-block mr-1 text-red-600" />;
  }
  return <MinusCircle className="w-4 h-4 inline-block mr-1 text-gray-500" />; // Default for HOLD etc.
}

export default function ReasoningDisplay({ preloadedData }: ReasoningDisplayProps) {
  const [reasoningData, setReasoningData] = useState<ReasoningItem[] | null>(preloadedData || null);
  const [isLoading, setIsLoading] = useState(!preloadedData);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // If preloaded data is provided, use it directly
    if (preloadedData) {
      setReasoningData(preloadedData);
      setIsLoading(false);
      return;
    }

    // Otherwise fetch data as usual
    async function fetchReasoningData() {
      setIsLoading(true);
      setError(null);
      try {
        const response = await fetch('/api/reasoning');
        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.error || `Failed to fetch reasoning data: ${response.statusText}`);
        }
        const data: ReasoningItem[] = await response.json();
        setReasoningData(data);
      } catch (err: unknown) {
        console.error("Error fetching reasoning:", err);
        let message = 'An unexpected error occurred while fetching reasoning data.';
        if (err instanceof Error) {
          message = err.message;
        }
        setError(message);
      }
      setIsLoading(false);
    }

    fetchReasoningData();
  }, [preloadedData]);

  const renderLoading = () => (
    <div className="text-center p-4 md:p-10 bg-white rounded-lg shadow-md">
      <p className="text-gray-500">Loading AI reasoning data...</p>
    </div>
  );

  const renderError = () => (
    <div className="text-center p-4 md:p-10 bg-red-50 rounded-lg shadow-md border border-red-200">
      <p className="text-red-700 font-semibold">Error Loading Reasoning Data</p>
      <p className="text-red-600 text-sm mt-2">{error}</p>
    </div>
  );

  const renderNoData = () => (
    <div className="text-center p-4 md:p-10 bg-white rounded-lg shadow-md">
       <p className="text-gray-500">No recent AI reasoning data available.</p>
    </div>
  );

  if (isLoading) return renderLoading();
  if (error) return renderError();
  if (!reasoningData || reasoningData.length === 0) return renderNoData();

  // Group data by ticker
  const groupedByTicker = reasoningData.reduce((acc, item) => {
    const ticker = item.ticker;
    if (!acc[ticker]) {
      acc[ticker] = [];
    }
    acc[ticker].push(item);
    return acc;
  }, {} as Record<string, ReasoningItem[]>);

  return (
    <section className="bg-white shadow-lg rounded-lg p-4 md:p-8 mt-6 md:mt-8">
      <h2 className="text-xl md:text-2xl font-semibold mb-4 md:mb-6 text-gray-700 border-b pb-3">Recent AI Trading Analysis</h2>
      <div className="space-y-6 md:space-y-8"> {/* Outer container for ticker groups */}
        {Object.entries(groupedByTicker).map(([ticker, items]) => (
          <div key={ticker} className="ticker-group"> {/* Wrapper for each ticker's section */}
            <h3 className="text-lg md:text-xl font-semibold text-indigo-700 mb-3 md:mb-4 sticky top-0 bg-white py-2 z-10 border-b border-indigo-100 capitalize">{ticker}</h3>
            <div className="space-y-4 md:space-y-6"> {/* Container for items within this ticker group */}
              {items.map((item) => (
                <div key={item.id} className="p-3 md:p-5 border border-gray-200 rounded-lg hover:shadow-lg transition-shadow bg-gray-50/50">
                  <div className="flex flex-col justify-between items-start mb-3">
                    <div className="mb-2 flex flex-wrap items-center gap-1.5 md:gap-2"> {/* Added flex-wrap and gap for better spacing */}
                      <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-2xs md:text-xs font-medium 
                        ${item.action.toUpperCase().includes('BUY') || item.action.toUpperCase().includes('COVER') ? 'bg-green-100 text-green-800' : 
                          item.action.toUpperCase().includes('SELL') || item.action.toUpperCase().includes('SHORT') ? 'bg-red-100 text-red-800' : 'bg-gray-100 text-gray-800'}
                      `}>
                        {getActionIcon(item.action)} {item.action.toUpperCase()}
                      </span>

                      {/* Differentiate Stock vs Option Analysis */}
                      <span className={`px-2 py-0.5 rounded-full text-2xs md:text-xs font-medium 
                        ${item.item_type === 'Stock' ? 'bg-sky-100 text-sky-800' : 'bg-amber-100 text-amber-800'}
                      `}>
                        {item.item_type}
                      </span>

                      {item.item_type === 'Option' && item.option_ticker && (
                         <span className="text-2xs md:text-xs font-mono text-blue-600 bg-blue-50 px-1.5 py-0.5 rounded">{item.option_ticker}</span>
                      )}
                       {item.item_type === 'Option' && item.strategy && (
                         <span className="text-2xs md:text-xs font-medium text-purple-700">({item.strategy})</span>
                      )}
                    </div>
                    <span className="text-2xs md:text-xs text-gray-500 flex-shrink-0">{new Date(item.timestamp).toLocaleString()}</span>
                  </div>
                  <p className="text-xs md:text-sm text-gray-700 mb-3 leading-relaxed pl-1 border-l-2 border-blue-200 ml-1">
                    {item.reasoning_summary}
                  </p>
                  <div className="flex flex-wrap gap-x-3 md:gap-x-4 gap-y-1 text-2xs md:text-xs text-gray-600">
                    {item.confidence != null && (
                      <span className="inline-flex items-center"><BarChart className="w-3 md:w-3.5 h-3 md:h-3.5 mr-1 text-indigo-500"/> Confidence: {item.confidence.toFixed(1)}%</span>
                    )}
                    {item.analysts_involved && item.analysts_involved.length > 0 && (
                       <span className="inline-flex items-center"><Users className="w-3 md:w-3.5 h-3 md:h-3.5 mr-1 text-teal-500"/> Analysts: {item.analysts_involved.join(', ')}</span>
                    ) }
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
} 