'use client';

import { useState, useEffect } from 'react';

interface Position {
  ticker: string;
  quantity: number;
  entry_price: number;
  current_price: number;
  market_value: number;
  unrealized_pl: number;
  unrealized_pl_pct: number;
}

interface PortfolioData {
  portfolio_value: number;
  cash_balance: number;
  equity: number;
  buying_power: number;
  positions: Position[];
  timestamp: string;
}

export default function PortfolioDisplay() {
  const [portfolioData, setPortfolioData] = useState<PortfolioData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchPortfolioData() {
      setIsLoading(true);
      setError(null);
      try {
        const response = await fetch('/api/portfolio');
        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.error || `Failed to fetch portfolio data: ${response.statusText}`);
        }
        const data: PortfolioData = await response.json();
        setPortfolioData(data);
      } catch (err: any) {
        console.error("Error fetching portfolio:", err);
        setError(err.message || 'An unexpected error occurred while fetching portfolio data.');
      }
      setIsLoading(false);
    }

    fetchPortfolioData();
  }, []);

  const renderLoading = () => (
    <div className="text-center p-10 bg-white rounded-lg shadow-md">
      <p className="text-gray-500">Loading portfolio data...</p>
      {/* Optional: Add a spinner SVG or component here */}
    </div>
  );

  const renderError = () => (
    <div className="text-center p-10 bg-red-50 rounded-lg shadow-md border border-red-200">
      <p className="text-red-700 font-semibold">Error Loading Portfolio</p>
      <p className="text-red-600 text-sm mt-2">{error}</p>
    </div>
  );

  const renderNoData = () => (
    <div className="text-center p-10 bg-white rounded-lg shadow-md">
      <p className="text-gray-500">No portfolio data available.</p>
    </div>
  );

  if (isLoading) return renderLoading();
  if (error) return renderError();
  if (!portfolioData) return renderNoData();

  // Helper to format currency
  const formatCurrency = (value: number | null | undefined) => {
    if (value === null || value === undefined) return 'N/A';
    return value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  };

  return (
    <section className="bg-white shadow-lg rounded-lg p-6 md:p-8">
      <h2 className="text-2xl font-semibold mb-6 text-gray-700 border-b pb-3">Live Paper Trading Portfolio</h2>
      
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 md:gap-6 mb-8 text-center">
        {/* Summary Cards - slightly more padding */}
        <div className="p-4 bg-blue-50 rounded-lg shadow-sm">
          <p className="text-sm text-blue-700 font-medium">Portfolio Value</p>
          <p className="text-xl font-bold text-blue-900">${formatCurrency(portfolioData.portfolio_value)}</p>
        </div>
        <div className="p-4 bg-green-50 rounded-lg shadow-sm">
          <p className="text-sm text-green-700 font-medium">Cash Balance</p>
          <p className="text-xl font-bold text-green-900">${formatCurrency(portfolioData.cash_balance)}</p>
        </div>
        <div className="p-4 bg-indigo-50 rounded-lg shadow-sm">
          <p className="text-sm text-indigo-700 font-medium">Total Equity</p>
          <p className="text-xl font-bold text-indigo-900">${formatCurrency(portfolioData.equity)}</p>
        </div>
        <div className="p-4 bg-purple-50 rounded-lg shadow-sm">
          <p className="text-sm text-purple-700 font-medium">Buying Power</p>
          <p className="text-xl font-bold text-purple-900">${formatCurrency(portfolioData.buying_power)}</p>
        </div>
      </div>

      <h3 className="text-xl font-semibold mb-4 text-gray-800">Current Positions</h3>
      {portfolioData.positions && portfolioData.positions.length > 0 ? (
        <div className="overflow-x-auto border border-gray-200 rounded-lg">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-100">
              <tr>
                {['Ticker', 'Qty', 'Entry Price', 'Current Price', 'Market Value', 'Unrealized P/L', 'P/L %'].map(header => (
                  <th key={header} scope="col" className="px-5 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    {header}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {portfolioData.positions.map((pos) => (
                <tr key={pos.ticker} className="hover:bg-gray-50 transition-colors duration-150 ease-in-out">
                  <td className="px-5 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{pos.ticker}</td>
                  <td className="px-5 py-4 whitespace-nowrap text-sm text-gray-600 text-right">{pos.quantity}</td>
                  <td className="px-5 py-4 whitespace-nowrap text-sm text-gray-600 text-right">${pos.entry_price?.toFixed(2)}</td>
                  <td className="px-5 py-4 whitespace-nowrap text-sm text-gray-600 text-right">${pos.current_price?.toFixed(2)}</td>
                  <td className="px-5 py-4 whitespace-nowrap text-sm text-gray-600 text-right">${formatCurrency(pos.market_value)}</td>
                  <td className={`px-5 py-4 whitespace-nowrap text-sm font-semibold text-right ${pos.unrealized_pl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {pos.unrealized_pl >= 0 ? '+' : ''}${formatCurrency(pos.unrealized_pl)}
                  </td>
                  <td className={`px-5 py-4 whitespace-nowrap text-sm font-semibold text-right ${pos.unrealized_pl_pct >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {pos.unrealized_pl_pct >= 0 ? '+' : ''}{pos.unrealized_pl_pct?.toFixed(2)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="text-center py-6 bg-gray-50 rounded-md border">
          <p className="text-gray-500 italic">No open positions.</p>
        </div>
      )}
      <p className="text-xs text-gray-400 mt-6 text-right">Last updated: {new Date(portfolioData.timestamp).toLocaleString()}</p>
    </section>
  );
} 