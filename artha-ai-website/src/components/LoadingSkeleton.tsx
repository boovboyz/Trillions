'use client';

import React from 'react';

// Common skeleton component for uniform loading behavior
export function SkeletonLoader({ className = "" }: { className?: string }) {
  return (
    <div className={`animate-pulse bg-gray-700 rounded-md ${className}`}></div>
  );
}

// Portfolio section skeleton
export function PortfolioSkeleton() {
  return (
    <section className="bg-white shadow-lg rounded-lg p-4 md:p-8 h-full w-full">
      <div className="h-8 mb-6 w-3/4 max-w-xs">
        <SkeletonLoader className="h-full w-full" />
      </div>
      
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 md:gap-6 mb-6 md:mb-8">
        {Array(4).fill(0).map((_, i) => (
          <div key={i} className="p-4 rounded-lg bg-gray-100 h-20">
            <div className="h-3 mb-2 w-2/3 mx-auto">
              <SkeletonLoader className="h-full w-full" />
            </div>
            <div className="h-6 w-4/5 mx-auto">
              <SkeletonLoader className="h-full w-full" />
            </div>
          </div>
        ))}
      </div>
      
      <div className="h-7 mb-4 w-48">
        <SkeletonLoader className="h-full w-full" />
      </div>
      
      <div className="h-60 w-full rounded-lg overflow-hidden">
        <SkeletonLoader className="h-full w-full" />
      </div>
    </section>
  );
}

// Trades section skeleton
export function TradesSkeleton() {
  return (
    <section className="bg-white shadow-lg rounded-lg p-4 md:p-8 h-full w-full">
      <div className="h-8 mb-6 w-1/3 max-w-xs">
        <SkeletonLoader className="h-full w-full" />
      </div>
      
      <div className="h-40 w-full rounded-lg overflow-hidden">
        <SkeletonLoader className="h-full w-full" />
      </div>
    </section>
  );
}

// Reasoning section skeleton
export function ReasoningSkeleton() {
  return (
    <section className="bg-white shadow-lg rounded-lg p-4 md:p-8 h-full w-full">
      <div className="h-8 mb-6 w-1/2 max-w-sm">
        <SkeletonLoader className="h-full w-full" />
      </div>
      
      <div className="space-y-4">
        {Array(3).fill(0).map((_, i) => (
          <div key={i} className="space-y-2">
            <div className="h-5 w-1/4 mb-2">
              <SkeletonLoader className="h-full w-full" />
            </div>
            <div className="h-24 w-full rounded-lg overflow-hidden">
              <SkeletonLoader className="h-full w-full" />
            </div>
          </div>
        ))}
      </div>
    </section>
  );
} 