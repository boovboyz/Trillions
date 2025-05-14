'use client';

import { useState, useEffect } from 'react';

interface Trade {
  id: string;
  ticker: string;
  type: string;
  quantity: number;
  price: number;
  timestamp: string;
  status: string;
}

export default function RecentTradesDisplay() {
  const [trades, setTrades] = useState<Trade[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchRecentTrades() {
      setIsLoading(true);
      setError(null);
      try {
        const response = await fetch('/api/trades');
        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.error || `Failed to fetch trades: ${response.statusText}`);
        }
        const data: Trade[] = await response.json();
        setTrades(data);
      } catch (err: unknown) {
        console.error("Error fetching recent trades:", err);
        let message = 'An unexpected error occurred while fetching trades data.';
        if (err instanceof Error) {
          message = err.message;
        }
        setError(message);
      }
      setIsLoading(false);
    }

    fetchRecentTrades();
  }, []);

  const renderLoading = () => (
    <div className="text-center p-4 md:p-10 bg-white rounded-lg shadow-md">
      <p className="text-gray-500">Loading recent trades...</p>
    </div>
  );

  const renderError = () => (
    <div className="text-center p-4 md:p-10 bg-red-50 rounded-lg shadow-md border border-red-200">
      <p className="text-red-700 font-semibold">Error Loading Trades</p>
      <p className="text-red-600 text-sm mt-2">{error}</p>
    </div>
  );

  const renderNoData = () => (
    <div className="text-center p-4 md:p-10 bg-white rounded-lg shadow-md">
      <p className="text-gray-500">No recent trades available.</p>
    </div>
  );

  if (isLoading) return renderLoading();
  if (error) return renderError();
  if (!trades || trades.length === 0) return renderNoData();

  // Helper to format date
  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleString();
  };

  return (
    <section className="bg-white shadow-lg rounded-lg p-4 md:p-8">
      <h2 className="text-xl md:text-2xl font-semibold mb-4 md:mb-6 text-gray-700 border-b pb-3">Recent Trades</h2>
      
      <div className="overflow-x-auto -mx-4 px-4 pb-2">
        <div className="border border-gray-200 rounded-lg min-w-full">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-100">
              <tr>
                <th scope="col" className="px-2 md:px-5 py-2 md:py-3 text-left text-2xs md:text-xs font-medium text-gray-500 uppercase tracking-wider">Ticker</th>
                <th scope="col" className="px-2 md:px-5 py-2 md:py-3 text-left text-2xs md:text-xs font-medium text-gray-500 uppercase tracking-wider">Type</th>
                <th scope="col" className="px-2 md:px-5 py-2 md:py-3 text-right text-2xs md:text-xs font-medium text-gray-500 uppercase tracking-wider">Qty</th>
                <th scope="col" className="px-2 md:px-5 py-2 md:py-3 text-right text-2xs md:text-xs font-medium text-gray-500 uppercase tracking-wider">Price</th>
                <th scope="col" className="px-2 md:px-5 py-2 md:py-3 text-left text-2xs md:text-xs font-medium text-gray-500 uppercase tracking-wider">Date</th>
                <th scope="col" className="px-2 md:px-5 py-2 md:py-3 text-left text-2xs md:text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {trades.map((trade) => (
                <tr key={trade.id} className="hover:bg-gray-50 transition-colors duration-150 ease-in-out">
                  <td className="px-2 md:px-5 py-2 md:py-4 whitespace-nowrap text-xs md:text-sm font-medium text-gray-900">{trade.ticker}</td>
                  <td className={`px-2 md:px-5 py-2 md:py-4 whitespace-nowrap text-xs md:text-sm font-semibold 
                    ${trade.type.toLowerCase() === 'buy' ? 'text-green-600' : 'text-red-600'}`}>
                    {trade.type.toUpperCase()}
                  </td>
                  <td className="px-2 md:px-5 py-2 md:py-4 whitespace-nowrap text-xs md:text-sm text-gray-600 text-right">{trade.quantity}</td>
                  <td className="px-2 md:px-5 py-2 md:py-4 whitespace-nowrap text-xs md:text-sm text-gray-600 text-right">${trade.price.toFixed(2)}</td>
                  <td className="px-2 md:px-5 py-2 md:py-4 whitespace-nowrap text-xs md:text-sm text-gray-600">{formatDate(trade.timestamp)}</td>
                  <td className="px-2 md:px-5 py-2 md:py-4 whitespace-nowrap text-xs md:text-sm text-gray-600 capitalize">{trade.status}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
} 