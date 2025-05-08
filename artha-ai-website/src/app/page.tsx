'use client';

import { useState } from 'react';
import Link from 'next/link';
// Import icons from lucide-react
import { TrendingUp, ShieldCheck, MessageSquareText, BrainCircuit, Bot, Network, Scale, Activity, RefreshCw } from 'lucide-react';
import PortfolioDisplay from '@/components/PortfolioDisplay';
import ReasoningDisplay from '@/components/ReasoningDisplay';
import ContactForm from '@/components/ContactForm';

const features = [
  {
    icon: TrendingUp, // Icon for profit generation
    title: "Intelligent Profit Generation",
    description: "Leveraging deep market insights and predictive analytics to identify high-potential opportunities beyond simple trends.",
    iconBgColor: "bg-green-100",
    iconTextColor: "text-green-600",
  },
  {
    icon: ShieldCheck, // Icon for risk management
    title: "Advanced Risk Shield",
    description: "Sophisticated protocols, including dynamic stop-losses and position sizing, continuously work to safeguard your capital.",
    iconBgColor: "bg-red-100",
    iconTextColor: "text-red-600",
  },
  {
    icon: MessageSquareText, // Icon for transparency/reasoning
    title: "Transparent AI Reasoning",
    description: "Gain confidence with clear explanations for every trade, understanding the \"why\" behind the AI's strategic decisions.",
    iconBgColor: "bg-yellow-100",
    iconTextColor: "text-yellow-700",
  },
  {
    icon: BrainCircuit, // Icon for multi-analyst AI engine
    title: "Multi-Analyst Engine",
    description: "Benefit from a diverse team of specialized AI analysts, ensuring robust, well-rounded decision-making.",
    iconBgColor: "bg-purple-100",
    iconTextColor: "text-purple-600",
  },
];

export default function HomePage() {
  // State to trigger refresh
  const [refreshKey, setRefreshKey] = useState(0);

  const handleRefresh = () => {
    setRefreshKey(prevKey => prevKey + 1);
  };

  return (
    <main className="flex flex-col items-center justify-center">
      {/* Hero Section */}
      <section id="hero" className="w-full min-h-screen flex items-center justify-center py-20 md:py-32 bg-gradient-to-br from-gray-900 via-gray-800 to-black text-white text-center relative overflow-hidden">
        {/* Optional: Add subtle background pattern or animation here */}
        <div className="container mx-auto px-4 md:px-6 space-y-10 z-10">
          <h1 className="text-4xl font-extrabold tracking-tight sm:text-6xl md:text-7xl lg:text-8xl">
            AI-Powered Trading.
            <span className="block text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-teal-400 to-green-400 mt-2 sm:mt-4">Intelligently Optimized.</span>
          </h1>
          <p className="max-w-2xl mx-auto text-lg text-gray-300 sm:text-xl md:text-2xl leading-relaxed">
            ARTHA AI leverages a sophisticated multi-analyst engine to maximize your investment potential while actively managing risk. Experience the future of automated trading.
          </p>
          <div className="flex flex-col sm:flex-row justify-center items-center space-y-4 sm:space-y-0 sm:space-x-6">
            {/* Remove onClick, rely on CSS smooth scroll */}
            <a 
              href="#demo"
              onClick={(e) => {e.preventDefault(); document.getElementById('demo')?.scrollIntoView({ behavior: 'smooth' });}}
              className="inline-flex items-center justify-center px-10 py-4 text-xl font-semibold text-white bg-gradient-to-r from-blue-600 to-teal-500 rounded-lg shadow-xl hover:from-blue-700 hover:to-teal-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-gray-900 transition-all transform hover:scale-105 duration-300 ease-in-out"
            >
              See Live Demo
            </a>
            {/* Remove onClick, rely on CSS smooth scroll */}
            <a 
              href="#features"
              onClick={(e) => {e.preventDefault(); document.getElementById('features')?.scrollIntoView({ behavior: 'smooth' });}} 
              className="inline-flex items-center justify-center px-10 py-4 text-xl font-semibold text-blue-300 bg-transparent border-2 border-blue-400 rounded-lg shadow-md hover:bg-blue-900/30 hover:text-white hover:border-blue-300 focus:outline-none focus:ring-2 focus:ring-blue-400 focus:ring-offset-2 focus:ring-offset-gray-900 transition-all transform hover:scale-105 duration-300 ease-in-out"
            >
              Explore Features
            </a>
          </div>
        </div>
      </section>

      {/* Features Section (Previously "Why Choose") */}
      <section id="features" className="w-full py-16 md:py-24 bg-white">
        <div className="container mx-auto px-4 md:px-6">
          <div className="text-center mb-12 md:mb-16">
            <h2 className="text-3xl sm:text-4xl font-extrabold text-gray-900 tracking-tight">
              The ARTHA AI Advantage
            </h2>
            <p className="mt-4 max-w-2xl mx-auto text-lg sm:text-xl text-gray-600">
              Combining cutting-edge AI with robust financial principles for superior trading outcomes.
            </p>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-8">
            {features.map((feature) => (
              <div key={feature.title} className="bg-gray-50/50 p-6 rounded-xl shadow-lg border border-gray-100 hover:shadow-xl transition-shadow duration-300 flex flex-col items-start text-left">
                <div className={`mb-4 p-3 rounded-full ${feature.iconBgColor} inline-block`}>
                  <feature.icon className={`w-8 h-8 ${feature.iconTextColor}`} />
                </div>
                <h3 className="mb-2 text-xl font-semibold text-gray-900">{feature.title}</h3>
                <p className="text-gray-600 text-base leading-relaxed flex-grow">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section id="how-it-works" className="w-full py-16 md:py-24 bg-gray-50">
        <div className="container mx-auto px-4 md:px-6">
          <div className="text-center mb-12 md:mb-16">
             <h2 className="text-3xl sm:text-4xl font-extrabold text-gray-900 tracking-tight">
              Inside the Intelligent Engine
            </h2>
             <p className="mt-4 max-w-2xl mx-auto text-lg sm:text-xl text-gray-600">
              Discover the sophisticated process behind ARTHA AI's market analysis and decision-making.
            </p>
          </div>
          {/* Adapted content from how-it-works page */}
          <div className="max-w-4xl mx-auto grid grid-cols-1 md:grid-cols-2 gap-10 lg:gap-16 items-start">
             <div className="space-y-6 bg-white p-6 rounded-lg shadow-md border">
                <h3 className="flex items-center text-2xl font-bold text-gray-800"><Network className="w-7 h-7 mr-3 text-blue-600"/> The Multi-Analyst Engine™</h3>
                <p className="text-gray-700 leading-relaxed">
                  Instead of a single algorithm, ARTHA AI simulates a diverse team of specialized AI analysts (Value, Growth, Technical, Sentiment, etc.). Each provides unique insights, weighted and synthesized for a robust, consensus-driven decision.
                </p>
             </div>
              <div className="space-y-6 bg-white p-6 rounded-lg shadow-md border">
                <h3 className="flex items-center text-2xl font-bold text-gray-800"><Scale className="w-7 h-7 mr-3 text-red-600"/> Risk Management Framework</h3>
                <p className="text-gray-700 leading-relaxed">
                  Capital preservation is key. We employ dynamic position sizing, automated stop-losses, profit-taking rules, and market condition awareness to actively manage downside risk.
                </p>
             </div>
              <div className="space-y-6 bg-white p-6 rounded-lg shadow-md border">
                <h3 className="flex items-center text-2xl font-bold text-gray-800"><Bot className="w-7 h-7 mr-3 text-purple-600"/> Transparent Reasoning</h3>
                <p className="text-gray-700 leading-relaxed">
                   Our AI doesn't operate in a black box. View clear summaries explaining the factors driving each trading decision, providing crucial insight into the strategy.
                </p>
             </div>
             <div className="space-y-6 bg-white p-6 rounded-lg shadow-md border">
                <h3 className="flex items-center text-2xl font-bold text-gray-800"><Activity className="w-7 h-7 mr-3 text-teal-600"/> Continuous Adaptation</h3>
                <p className="text-gray-700 leading-relaxed">
                  Markets evolve. ARTHA AI incorporates continuous learning mechanisms, adapting its models and strategies based on new data and performance feedback.
                </p>
             </div>
          </div>
        </div>
      </section>

      {/* Live Demo Section */}
      <section id="demo" className="w-full py-16 md:py-24 bg-gradient-to-br from-gray-900 via-gray-800 to-black text-white">
         <div className="container mx-auto px-4 md:px-6">
             <div className="text-center mb-12 md:mb-16 relative">
                <h2 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight">
                    ARTHA AI Live Demo
                </h2>
                <p className="mt-4 max-w-2xl mx-auto text-lg sm:text-xl text-gray-300">
                    See a snapshot of our AI's paper trading activity and decision reasoning in near real-time.
                </p>
                 {/* Refresh Button */}
                <button 
                  onClick={handleRefresh}
                  className="absolute top-0 right-0 mt-2 mr-2 sm:mt-0 sm:mr-0 inline-flex items-center px-3 py-1.5 border border-transparent text-xs font-medium rounded-full shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-800 focus:ring-blue-500 transition-colors"
                  title="Refresh Demo Data"
                >
                    <RefreshCw className="w-4 h-4 mr-1" /> Refresh
                </button>
             </div>
             {/* Embed Portfolio and Reasoning Displays with Key Prop */}
             <div className="space-y-12 max-w-6xl mx-auto">
                <PortfolioDisplay key={`portfolio-${refreshKey}`} />
                <ReasoningDisplay key={`reasoning-${refreshKey}`} />
             </div>
             <p className="text-center text-xs text-gray-400 mt-12 max-w-3xl mx-auto">
                Disclaimer: The demo utilizes a paper trading account and may include historical or simulated data for illustrative purposes. Trading involves substantial risk, and past performance is not indicative of future results.
            </p>
         </div>
      </section>

      {/* Contact Section */}
      <section id="contact" className="w-full py-16 md:py-24 bg-gray-100">
        <div className="container mx-auto px-4 md:px-6">
          <div className="max-w-xl mx-auto text-center mb-12 md:mb-16">
            <h2 className="text-3xl sm:text-4xl font-extrabold text-gray-900 tracking-tight">
              Ready to Elevate Your Trading?
            </h2>
            <p className="mt-4 text-lg sm:text-xl text-gray-600">
              Reach out to learn more about ARTHA AI or discuss potential investment opportunities.
            </p>
          </div>
          <div className="max-w-2xl mx-auto">
             <ContactForm />
          </div>
    </div>
      </section>

    </main>
  );
}
