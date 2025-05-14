'use client';

import React, { useState, useEffect } from 'react';
import { RefreshCw } from 'lucide-react';
import PortfolioDisplay from './PortfolioDisplay';
import ReasoningDisplay from './ReasoningDisplay';
import RecentTradesDisplay from './RecentTradesDisplay';
import { PortfolioSkeleton, TradesSkeleton, ReasoningSkeleton } from './LoadingSkeleton';

export default function LiveDemoContainer() {
  const [refreshKey, setRefreshKey] = useState<number>(0);
  const [isLoading, setIsLoading] = useState(true);
  
  // State for preloaded data
  const [portfolioData, setPortfolioData] = useState<any>(null);
  const [tradesData, setTradesData] = useState<any>(null);
  const [reasoningData, setReasoningData] = useState<any>(null);

  // Handle data prefetching to reduce jitter during refresh
  useEffect(() => {
    async function fetchAllData() {
      setIsLoading(true);
      
      try {
        // Fetch all data in parallel for better performance
        const [portfolioResponse, tradesResponse, reasoningResponse] = await Promise.all([
          fetch('/api/portfolio'),
          fetch('/api/trades'),
          fetch('/api/reasoning')
        ]);

        // Process responses only if they're all successful
        if (portfolioResponse.ok && tradesResponse.ok && reasoningResponse.ok) {
          const portfolioData = await portfolioResponse.json();
          const tradesData = await tradesResponse.json();
          const reasoningData = await reasoningResponse.json();
          
          // Store fetched data in state
          setPortfolioData(portfolioData);
          setTradesData(tradesData);
          setReasoningData(reasoningData);
        }
      } catch (error) {
        console.error("Error fetching demo data:", error);
      } finally {
        // Short timeout to ensure smooth transition
        setTimeout(() => {
          setIsLoading(false);
        }, 300);
      }
    }

    fetchAllData();
  }, [refreshKey]);

  const handleRefresh = () => {
    setIsLoading(true);
    setRefreshKey((prevKey) => prevKey + 1);
  };

  return (
    <div className="container mx-auto px-4 md:px-6 relative z-10">
      <div className="text-center mb-8 md:mb-10 relative">
        <h2 className="text-2xl sm:text-3xl md:text-4xl lg:text-5xl font-extrabold text-white tracking-tight font-mono enhanced-glow-animated">
          VectorQuant AI Live Demo
        </h2>
        <p className="mt-4 max-w-2xl mx-auto text-base md:text-lg text-gray-300">
          Witness VectorQuant AI in action! This live demo showcases our AI&apos;s paper trading portfolio and decision reasoning in near real-time. The system actively manages a <strong>$1 Million portfolio</strong> with positions in <strong>AAPL, MSFT, NVDA, and TSLA</strong>, updating its operations <strong>every hour</strong>.
        </p>
        {/* Refresh Button */}
        <button 
          onClick={handleRefresh}
          disabled={isLoading}
          className={`mt-4 md:mt-0 md:absolute md:top-0 md:right-0 inline-flex items-center px-3 py-1.5 md:px-4 md:py-2 border border-transparent text-xs md:text-sm font-medium rounded-full shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-800 focus:ring-indigo-500 transition-colors ${isLoading ? 'opacity-70 cursor-not-allowed' : 'pulse-glow'}`}
          title="Refresh Demo Data"
        >
          <RefreshCw className={`w-3.5 h-3.5 md:w-4 md:h-4 mr-1.5 md:mr-2 ${isLoading ? 'animate-spin' : ''}`} /> 
          {isLoading ? 'Loading...' : 'Refresh Data'}
        </button>
      </div>
      
      {/* Demo display with enhanced glow effect */}
      <div className="relative p-1 max-w-6xl mx-auto rounded-xl bg-gradient-to-r from-indigo-500 via-cyan-500 to-pink-500 shadow-3d enhanced-glow-animated">
        <div className="absolute inset-0 bg-gradient-to-r from-indigo-500 via-cyan-500 to-pink-500 opacity-70 blur-xl rounded-xl"></div>
        <div className="relative bg-gray-900 rounded-lg p-4 md:p-6 lg:p-8 space-y-8 md:space-y-12 shadow-inner backdrop-blur-sm">
          {/* Portfolio Display with skeleton fallback */}
          <div className="min-h-[300px]" style={{ contain: 'content' }}>
            {isLoading ? (
              <PortfolioSkeleton />
            ) : (
              <PortfolioDisplay preloadedData={portfolioData} />
            )}
          </div>
          
          {/* Trades Display with skeleton fallback */}
          <div className="min-h-[200px]" style={{ contain: 'content' }}>
            {isLoading ? (
              <TradesSkeleton />
            ) : (
              <RecentTradesDisplay preloadedData={tradesData} />
            )}
          </div>
          
          {/* Reasoning Display with skeleton fallback */}
          <div className="min-h-[400px]" style={{ contain: 'content' }}>
            {isLoading ? (
              <ReasoningSkeleton />
            ) : (
              <ReasoningDisplay preloadedData={reasoningData} />
            )}
          </div>
        </div>
      </div>
      
      <p className="text-center text-2xs md:text-xs text-gray-400 mt-6 md:mt-8 max-w-3xl mx-auto">
        Disclaimer: The demo utilizes a paper trading account and may include historical or simulated data for illustrative purposes. Trading involves substantial risk, and past performance is not indicative of future results.
      </p>
    </div>
  );
} 