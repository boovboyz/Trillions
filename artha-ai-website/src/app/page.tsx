'use client';

import React, { useState, useEffect } from 'react';
// Import icons from lucide-react
import { TrendingUp, ShieldCheck, MessageSquareText, BrainCircuit, Bot, Network, Scale, Activity, RefreshCw, ArrowRight, ChevronDown } from 'lucide-react';
import PortfolioDisplay from '@/components/PortfolioDisplay';
import ReasoningDisplay from '@/components/ReasoningDisplay';
import RecentTradesDisplay from '@/components/RecentTradesDisplay';
import ContactForm from '@/components/ContactForm';

const features = [
  {
    icon: TrendingUp, // Icon for profit generation
    title: "Intelligent Profit Generation",
    description: "By leveraging historical and real-time data with predictive analytics we can identify high-potential opportunities beyond simple trends.",
    iconBgColor: "bg-indigo-100",
    iconTextColor: "text-indigo-600",
    gradientClass: "from-indigo-500 to-indigo-700"
  },
  {
    icon: ShieldCheck, // Icon for risk management
    title: "Advanced Risk Shield",
    description: "Sophisticated protocols, including dynamic stop-losses and position sizing, continuously work to safeguard your capital while generating profits.",
    iconBgColor: "bg-pink-100",
    iconTextColor: "text-pink-600",
    gradientClass: "from-pink-500 to-pink-700"
  },
  {
    icon: MessageSquareText, // Icon for transparency/reasoning
    title: "Transparent AI Reasoning",
    description: "Gain confidence with clear explanations for every trade, understanding the \"why\" behind the AI's strategic decisions.",
    iconBgColor: "bg-cyan-100",
    iconTextColor: "text-cyan-700",
    gradientClass: "from-cyan-500 to-cyan-700"
  },
  {
    icon: BrainCircuit, // Icon for multi-analyst AI engine
    title: "Multi-Analyst Engine",
    description: "Benefit from a diverse team of specialized AI analysts, ensuring robust and well-rounded decision-making that intelligently generates profits.",
    iconBgColor: "bg-violet-100",
    iconTextColor: "text-violet-600",
    gradientClass: "from-violet-500 to-violet-700"
  },
];

export default function HomePage() {
  const [refreshKey, setRefreshKey] = useState<number>(0);
  const [scrollY, setScrollY] = useState<number>(0);
  // State for client-side calculated styles to avoid hydration mismatch
  const [multiAnalystNodeStyles, setMultiAnalystNodeStyles] = useState<React.CSSProperties[]>([]);
  const [continuousAdaptationBarStyles, setContinuousAdaptationBarStyles] = useState<React.CSSProperties[]>([]);
  // State for client-side generated arrows to avoid hydration mismatch
  const [heroArrows, setHeroArrows] = useState<React.ReactNode[]>([]);
  const [demoArrows, setDemoArrows] = useState<React.ReactNode[]>([]);

  useEffect(() => {
    const handleScroll = () => {
      setScrollY(window.scrollY);
    };
    window.addEventListener('scroll', handleScroll);

    // Calculate styles for Multi-Analyst Engine nodes on client
    const analystNodes = Array(5).fill(null).map((_, i) => ({
      transform: `translate(${Math.cos(i * Math.PI * 0.4) * 100}px, ${Math.sin(i * Math.PI * 0.4) * 60}px)`,
      animationDelay: `${i * 0.2}s`
    }));
    setMultiAnalystNodeStyles(analystNodes);

    // Calculate styles for Continuous Adaptation bars on client
    const adaptationBars = Array(20).fill(null).map((_, i) => ({
      height: `${Math.max(5, Math.sin(i * 0.5) * 30 + Math.random() * 20 + 10)}px`,
      animationDelay: `${i * 0.1}s`
    }));
    setContinuousAdaptationBarStyles(adaptationBars);

    // Generate hero section arrows client-side
    const generateHeroArrows = () => {
      return Array.from({ length: 60 }).map((_, index) => {
        // All arrows point up
        const isUp = true;
        const size = Math.random() * 1.2 + 0.8;
        const position = Math.random() * 100; // horizontal position
        
        // Distribute arrows vertically throughout the viewport
        const startPosition = Math.random() * 120; // 0-120% of viewport height
        
        // Vary the duration slightly
        const duration = Math.random() * 8 + 12; // 12-20s
        
        // Calculate delay to create a continuous flow effect
        // Arrows starting higher up should be further along in their animation
        const delay = (startPosition / 120) * duration * -1;
        
        const type = Math.random() > 0.6 
          ? Math.random() > 0.5 
            ? 'arrow-red' 
            : 'arrow-green'
          : Math.random() > 0.5
            ? 'arrow-blue'
            : 'arrow-purple';
              
        return (
          <div 
            key={`arrow-${index}`}
            className={`stock-arrow arrow-up ${type}`}
            style={{
              left: `${position}%`,
              scale: size,
              bottom: `${-80 + startPosition}vh`, // Adjust start position
              animationDuration: `${duration}s`,
              animationDelay: `${delay}s`,
              zIndex: 0
            }}
          ></div>
        );
      });
    };
    
    // Generate demo section arrows client-side
    const generateDemoArrows = () => {
      return Array.from({ length: 80 }).map((_, index) => {
        // All arrows point up
        const isUp = true;
        const size = Math.random() * 1.2 + 0.8;
        const position = Math.random() * 100; // horizontal position
        
        // Distribute arrows vertically throughout the viewport
        const startPosition = Math.random() * 150; // 0-150% of viewport height
        
        // Vary the duration slightly but make them faster than hero section
        const duration = Math.random() * 6 + 10; // 10-16s for faster movement
        
        // Calculate delay based on vertical position to maintain continuous flow
        const delay = (startPosition / 150) * duration * -1;
        
        const type = Math.random() > 0.6 
          ? Math.random() > 0.5 
            ? 'arrow-red' 
            : 'arrow-green'
          : Math.random() > 0.5
            ? 'arrow-blue'
            : 'arrow-purple';
              
        return (
          <div 
            key={`demo-arrow-${index}`}
            className={`stock-arrow arrow-up ${type}`}
            style={{
              left: `${position}%`,
              scale: size,
              bottom: `${-80 + startPosition}vh`, // Adjust start position
              animationDuration: `${duration}s`,
              animationDelay: `${delay}s`,
              zIndex: 0
            }}
          ></div>
        );
      });
    };
    
    setHeroArrows(generateHeroArrows());
    setDemoArrows(generateDemoArrows());

    return () => window.removeEventListener('scroll', handleScroll);
  }, []); // Empty dependency array ensures this runs once on mount (client-side)

  const handleRefresh = () => {
    setRefreshKey((prevKey: number) => prevKey + 1);
  };

  const scrollToSection = (e: React.MouseEvent<HTMLAnchorElement>, id: string) => {
    e.preventDefault();
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <main className="flex flex-col items-center justify-center w-full overflow-hidden">
      {/* Hero Section */}
      <section id="hero" className="w-full min-h-screen flex items-center justify-center py-16 md:py-24 bg-gradient-to-bl from-indigo-950 via-gray-900 to-black text-white text-center relative overflow-hidden">
        {/* Background elements - enhanced with modern effects */}
        <div className="absolute inset-0 w-full h-full z-0 overflow-hidden">
          {/* Stock arrows animation background */}
          <div className="stock-arrows-bg">
            {/* Grid pattern background */}
            <div className="grid-pattern-bg"></div>
            
            {/* We'll create arrows programmatically with JavaScript in useEffect */}
            {heroArrows}
          </div>
        </div>
        
        <div className="container mx-auto px-4 md:px-6 flex flex-col items-center justify-center z-10 max-w-4xl">
          <div className="text-center w-full mb-8 glass-hero-box p-10 rounded-xl">
            <h1 className="text-4xl font-semibold tracking-tight sm:text-6xl md:text-6xl lg:text-7xl whitespace-nowrap enhanced-glow-animated">
              Algorithmic Precision
              <span className="block gradient-text bg-gradient-to-r from-indigo-400 via-cyan-400 to-pink-400 mt-2 sm:mt-4 whitespace-nowrap font-extrabold">Analytical Performance</span>
            </h1>
            <p className="mt-8 mx-auto text-lg text-gray-100 sm:text-xl md:text-2xl leading-relaxed max-w-3xl font-normal antialiased">
              VectorQuant AI is an Advanced Full-Stack, Agentic AI system for Automated & Intelligent Investing & Portfolio Management<br /><span className="font-bold gradient-text bg-gradient-to-r from-cyan-300 to-indigo-300">Maximize Potential, Control Risk, and Trade Smarter</span>
            </p>
          </div>
          
          <div className="flex flex-col sm:flex-row justify-center items-center space-y-4 sm:space-y-0 sm:space-x-6 mt-8">
            <a 
              href="#demo"
              onClick={(e: React.MouseEvent<HTMLAnchorElement>) => scrollToSection(e, 'demo')}
              className="group inline-flex items-center justify-center px-10 py-4 text-xl font-semibold text-white bg-gradient-to-r from-indigo-600 to-cyan-500 rounded-lg shadow-3d hover:from-indigo-700 hover:to-cyan-600 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 focus:ring-offset-gray-900 transition-all transform hover:scale-105 duration-300 ease-in-out w-full sm:w-auto enhanced-glow-animated"
            >
              <span>See Live Demo</span>
              <ArrowRight className="ml-2 h-5 w-5 transform group-hover:translate-x-1 transition-transform" />
            </a>
            <a 
              href="#features"
              onClick={(e: React.MouseEvent<HTMLAnchorElement>) => scrollToSection(e, 'features')} 
              className="inline-flex items-center justify-center px-10 py-4 text-xl font-semibold text-indigo-300 bg-transparent border-2 border-indigo-400 rounded-lg shadow-md hover:bg-indigo-900/30 hover:text-white hover:border-indigo-300 focus:outline-none focus:ring-2 focus:ring-indigo-400 focus:ring-offset-2 focus:ring-offset-gray-900 transition-all transform hover:scale-105 duration-300 ease-in-out w-full sm:w-auto backdrop-blur-sm"
            >
              Explore Features
            </a>
          </div>
          
          {/* Scroll indicator with enhanced glow */}
          <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2 animate-bounce">
            <a 
              href="#features" 
              onClick={(e: React.MouseEvent<HTMLAnchorElement>) => scrollToSection(e, 'features')}
              aria-label="Scroll down"
              className="enhanced-glow-animated"
            >
              <ChevronDown className="h-8 w-8 text-white/80" />
            </a>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="w-full py-20 md:py-28 bg-white">
        <div className="container mx-auto px-4 md:px-6">
          <div className="text-center mb-16">
            <h2 className="text-3xl sm:text-4xl md:text-5xl font-extrabold text-gray-900 tracking-tight">
              The VectorQuant AI <span className="gradient-text bg-gradient-to-r from-indigo-500 via-cyan-500 to-pink-500">Advantage</span>
            </h2>
            <p className="mt-4 max-w-2xl mx-auto text-lg sm:text-xl text-gray-600">
              Combining cutting-edge AI with robust financial principles and data for superior trading outcomes that compete with top HedgeFunds.
            </p>
          </div>
          
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-8">
            {features.map((feature, index) => (
              <div 
                key={feature.title} 
                className="card-3d bg-white p-8 rounded-xl shadow-3d border border-gray-200 transition-all duration-300 flex flex-col items-center text-center group hover:border-transparent"
                style={{animationDelay: `${index * 0.1}s`}}
              >
                <div className={`mb-4 p-3 rounded-full ${feature.iconBgColor} inline-block depth-effect group-hover:bg-gradient-to-br group-hover:${feature.gradientClass} transition-all duration-300`}>
                  <feature.icon className={`w-8 h-8 ${feature.iconTextColor} group-hover:text-white transition-colors duration-300`} />
                </div>
                <h3 className="mb-2 text-xl font-semibold text-gray-900">{feature.title}</h3>
                <p className="text-gray-600 text-base leading-relaxed flex-grow">{feature.description}</p>
                <div className="w-0 group-hover:w-full h-1 mt-4 bg-gradient-to-r from-indigo-500 via-cyan-500 to-pink-500 transition-all duration-500 rounded-full"></div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section id="how-it-works" className="w-full py-20 md:py-28 bg-white relative overflow-hidden">
        {/* Subtle background patterns */}
        <div className="absolute inset-0 gradient-mesh opacity-30"></div>
        <div className="absolute inset-0 gradient-dots opacity-10"></div>
        
        <div className="container mx-auto px-4 md:px-6 relative z-10">
          <div className="text-center mb-16">
             <h2 className="text-3xl sm:text-4xl md:text-5xl font-extrabold text-gray-900 tracking-tight">
              Inside the <span className="gradient-text bg-gradient-to-r from-indigo-500 via-cyan-500 to-pink-500">Intelligent Engine</span>
            </h2>
             <p className="mt-4 max-w-2xl mx-auto text-lg sm:text-xl text-gray-600">
              Discover the sophisticated process behind VectorQuant AI&apos;s market analysis and decision-making.
            </p>
          </div>
          
          {/* How it works content */}
          <div className="max-w-5xl mx-auto grid grid-cols-1 md:grid-cols-2 gap-12 lg:gap-16 items-start grid-auto-rows-fr">
             <div className="card-3d space-y-6 bg-white p-8 rounded-xl shadow-3d border border-gray-200 flex flex-col items-center text-center h-full subtle-glow hover:border-transparent backdrop-blur-sm bg-white/95">
                {/* 3D visualization placeholder */}
                <div className="relative h-40 bg-gradient-to-br from-indigo-50 to-indigo-100 rounded-lg overflow-hidden depth-effect w-full">
                  <div className="absolute inset-0 flex items-center justify-center">
                    {/* Connection lines from center to nodes - these render first as background */}
                    {multiAnalystNodeStyles.length > 0 && multiAnalystNodeStyles.map((style, i) => {
                      // Extract transform values to determine position
                      const transformMatch = style.transform ? style.transform.match(/translate\(([^,]+),\s*([^)]+)\)/) : null;
                      if (!transformMatch) return null;
                      
                      const xOffset = parseFloat(transformMatch[1]);
                      const yOffset = parseFloat(transformMatch[2]);
                      const distance = Math.sqrt(xOffset * xOffset + yOffset * yOffset);
                      const angle = Math.atan2(yOffset, xOffset) * (180 / Math.PI);
                      
                      return (
                        <div 
                          key={`connection-line-${i}`}
                          className="absolute h-0.5 bg-gradient-to-r from-indigo-300/60 to-cyan-300/30"
                          style={{
                            width: `${distance}px`,
                            transform: `rotate(${angle}deg)`,
                            transformOrigin: 'center left',
                            left: '50%',
                            top: '50%',
                            zIndex: 0
                          }}
                        ></div>
                      );
                    })}
                    
                    {/* Central brain node - render this on top of the lines */}
                    <div className="w-16 h-16 rounded-full bg-gradient-to-br from-indigo-500 to-indigo-600 shadow-3d flex items-center justify-center text-white z-10 relative">
                      <BrainCircuit className="w-8 h-8" />
                    </div>
                    
                    {/* Analyst nodes - also render on top of the lines */}
                    {multiAnalystNodeStyles.length > 0 && multiAnalystNodeStyles.map((style, i) => (
                      <div 
                        key={`analyst-node-${i}`}
                        className="absolute w-8 h-8 rounded-full bg-white shadow-md flex items-center justify-center z-10"
                        style={{
                          ...style, 
                          border: "1px solid rgba(79, 70, 229, 0.2)"
                        }}
                      >
                        <div className="w-4 h-4 rounded-full" style={{
                          background: `linear-gradient(to bottom right, 
                            ${['rgba(79, 70, 229, 0.8)', 'rgba(6, 182, 212, 0.8)', 'rgba(236, 72, 153, 0.8)', 'rgba(16, 185, 129, 0.8)', 'rgba(124, 58, 237, 0.8)'][i % 5]}, 
                            ${['rgba(79, 70, 229, 0.4)', 'rgba(6, 182, 212, 0.4)', 'rgba(236, 72, 153, 0.4)', 'rgba(16, 185, 129, 0.4)', 'rgba(124, 58, 237, 0.4)'][i % 5]})`
                        }}></div>
                      </div>
                    ))}
                  </div>
                </div>
                <h3 className="flex items-center justify-center text-2xl font-bold text-gray-800">
                  <Network className="w-7 h-7 mr-3 text-indigo-600"/> 
                  <span>The Multi-Analyst Engine™</span>
                </h3>
                <p className="text-gray-700 leading-relaxed">
                  Instead of a single algorithm, VectorQuant AI simulates a diverse team of specialized AI analysts (Value, Growth, Technical, Sentiment, etc.). Each provides unique insights, weighted and synthesized for a robust, consensus-driven decision.
                </p>
             </div>
              
             <div className="card-3d space-y-6 bg-white p-8 rounded-xl shadow-3d border border-gray-200 flex flex-col items-center text-center h-full subtle-glow hover:border-transparent backdrop-blur-sm bg-white/95">
                {/* 3D visualization placeholder */}
                <div className="relative h-40 bg-gradient-to-br from-pink-50 to-pink-100 rounded-lg overflow-hidden depth-effect w-full">
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="transform-gpu transition-transform animate-balance">
                      <Scale className="w-16 h-16 text-pink-600" />
                    </div>
                  </div>
                  <div className="absolute bottom-0 left-0 w-full h-1 bg-gradient-to-r from-pink-300 to-indigo-300"></div>
                </div>
                <h3 className="flex items-center justify-center text-2xl font-bold text-gray-800">
                  <Scale className="w-7 h-7 mr-3 text-pink-600"/> 
                  <span>Risk Management Framework</span>
                </h3>
                <p className="text-gray-700 leading-relaxed">
                  Capital preservation is key. We employ dynamic position sizing, automated stop-losses, profit-taking rules, and market condition awareness to actively manage downside risk and keep your money as safe as possible, while making a profit.
                </p>
             </div>
              
             <div className="card-3d space-y-6 bg-white p-8 rounded-xl shadow-3d border border-gray-200 flex flex-col items-center text-center h-full subtle-glow hover:border-transparent backdrop-blur-sm bg-white/95">
                {/* 3D visualization placeholder */}
                <div className="relative h-40 bg-gradient-to-br from-violet-50 to-violet-100 rounded-lg overflow-hidden depth-effect w-full">
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="w-3/4 h-24 bg-white rounded-lg shadow-md flex flex-col justify-center px-4">
                      <div className="w-1/2 h-2 bg-violet-200 rounded-full mb-2"></div>
                      <div className="w-3/4 h-2 bg-violet-200 rounded-full mb-2"></div>
                      <div className="w-2/3 h-2 bg-violet-200 rounded-full"></div>
                      <div className="absolute -right-2 -bottom-2 w-10 h-10 rounded-full bg-gradient-to-br from-violet-500 to-indigo-500 flex items-center justify-center text-white">
                        <Bot className="w-5 h-5" />
                      </div>
                    </div>
                  </div>
                </div>
                <h3 className="flex items-center justify-center text-2xl font-bold text-gray-800">
                  <Bot className="w-7 h-7 mr-3 text-violet-600"/> 
                  <span>Transparent Reasoning</span>
                </h3>
                <p className="text-gray-700 leading-relaxed">
                   Our AI doesn&apos;t operate in a black box. We believe in empowering our users with a clear understanding of the AI&apos;s decision-making process. View detailed summaries and key contributing factors that drive each trading decision. This crucial insight into the AI&apos;s strategy not only fosters trust but also allows for a more collaborative and informed investment experience, demystifying complex algorithmic actions.
                </p>
             </div>
             
             <div className="card-3d space-y-6 bg-white p-8 rounded-xl shadow-3d border border-gray-200 flex flex-col items-center text-center h-full subtle-glow hover:border-transparent backdrop-blur-sm bg-white/95">
                {/* 3D visualization placeholder */}
                <div className="relative h-40 bg-gradient-to-br from-cyan-50 to-cyan-100 rounded-lg overflow-hidden depth-effect w-full">
                  <div className="absolute inset-0 flex items-center justify-center overflow-hidden">
                    <div className="absolute h-40 w-full">
                      <div className="absolute bottom-0 left-0 w-full h-20 flex items-end">
                        {continuousAdaptationBarStyles.length > 0 && continuousAdaptationBarStyles.map((style, i) => (
                          <div 
                            key={`adaptation-bar-${i}`} 
                            className="w-3 bg-gradient-to-t from-cyan-600 to-cyan-300 mx-1 rounded-t opacity-80"
                            style={style}
                          ></div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
                <h3 className="flex items-center justify-center text-2xl font-bold text-gray-800">
                  <Activity className="w-7 h-7 mr-3 text-cyan-600"/> 
                  <span>Continuous Adaptation</span>
                </h3>
                <p className="text-gray-700 leading-relaxed">
                  As AI systems rapidly advance in power and sophistication, VectorQuant AI is designed to harness this exponential growth. Our continuous learning architecture ensures that the system not only adapts to new market data and performance feedback but also progressively enhances its predictive accuracy and strategic capabilities, growing more intelligent over time as foundational models get better.
                </p>
             </div>
          </div>
        </div>
      </section>

      {/* Live Demo Section */}
      <section id="demo" className="w-full py-20 md:py-28 bg-gradient-to-br from-indigo-950 via-gray-900 to-black text-white relative overflow-hidden">
        {/* Background effects */}
        <div className="stock-arrows-bg">
          {/* Grid pattern background */}
          <div className="grid-pattern-bg"></div>
          
          {/* Stock arrows */}
          {demoArrows}
        </div>
        
        <div className="container mx-auto px-4 md:px-6 relative z-10">
            <div className="text-center mb-10 relative">
               <h2 className="text-3xl sm:text-4xl md:text-5xl font-extrabold text-white tracking-tight font-mono enhanced-glow-animated">
                   VectorQuant AI Live Demo
               </h2>
               <p className="mt-4 max-w-2xl mx-auto text-lg sm:text-xl text-gray-300">
                   Witness VectorQuant AI in action! This live demo showcases our AI&apos;s paper trading portfolio and decision reasoning in near real-time. The system actively manages a <strong>$1 Million portfolio</strong> with positions in <strong>AAPL, MSFT, NVDA, and TSLA</strong>, updating its operations <strong>every hour</strong>.
               </p>
                {/* Refresh Button */}
               <button 
                 onClick={handleRefresh}
                 className="absolute top-0 right-0 mt-2 mr-2 sm:mt-0 sm:mr-0 inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-full shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-800 focus:ring-indigo-500 transition-colors pulse-glow"
                 title="Refresh Demo Data"
               >
                   <RefreshCw className="w-4 h-4 mr-2" /> Refresh Data
               </button>
            </div>
            
            {/* Demo display with enhanced glow effect */}
            <div className="relative p-1 max-w-6xl mx-auto rounded-xl bg-gradient-to-r from-indigo-500 via-cyan-500 to-pink-500 shadow-3d enhanced-glow-animated">
              <div className="absolute inset-0 bg-gradient-to-r from-indigo-500 via-cyan-500 to-pink-500 opacity-70 blur-xl rounded-xl"></div>
              <div className="relative bg-gray-900 rounded-lg p-6 md:p-8 space-y-12 shadow-inner backdrop-blur-sm">
                 <PortfolioDisplay key={`portfolio-${refreshKey}`} />
                 <RecentTradesDisplay key={`trades-${refreshKey}`} />
                 <ReasoningDisplay key={`reasoning-${refreshKey}`} />
              </div>
            </div>
            
            <p className="text-center text-xs text-gray-400 mt-8 max-w-3xl mx-auto">
               Disclaimer: The demo utilizes a paper trading account and may include historical or simulated data for illustrative purposes. Trading involves substantial risk, and past performance is not indicative of future results.
           </p>
        </div>
      </section>

      {/* Contact Section */}
      <section id="contact" className="w-full py-20 md:py-28 bg-white relative overflow-hidden">
        {/* Subtle background patterns */}
        <div className="absolute inset-0 gradient-mesh opacity-30"></div>
        <div className="absolute inset-0 gradient-dots opacity-10"></div>
        
        <div className="container mx-auto px-4 md:px-6 relative z-10">
          <div className="max-w-xl mx-auto text-center mb-16">
            <h2 className="text-3xl sm:text-4xl md:text-5xl font-extrabold text-gray-900 tracking-tight">
              Ready to <span className="gradient-text bg-gradient-to-r from-indigo-500 via-cyan-500 to-pink-500">Elevate</span> Your Trading?
            </h2>
            <p className="mt-4 text-lg sm:text-xl text-gray-600">
              Reach out to learn more about VectorQuant AI or discuss potential investment opportunities.
            </p>
          </div>
          
          <div className="relative max-w-2xl mx-auto">
            {/* 3D decorative elements */}
            <div className="absolute -top-10 -left-10 w-20 h-20 bg-indigo-200 rounded-full opacity-70 blur-xl animate-pulse-scale"></div>
            <div className="absolute -bottom-10 -right-10 w-20 h-20 bg-cyan-200 rounded-full opacity-70 blur-xl animate-pulse-scale" style={{ animationDelay: '1s' }}></div>
            <div className="absolute top-20 right-0 w-16 h-16 bg-pink-200 rounded-full opacity-60 blur-xl animate-pulse-scale" style={{ animationDelay: '2s' }}></div>
            
            {/* Card with glow effect */}
            <div className="relative p-1 rounded-xl bg-gradient-to-r from-indigo-500 via-cyan-500 to-pink-500 shadow-3d enhanced-glow-animated">
              <div className="bg-white rounded-lg p-6 md:p-8 shadow-inner backdrop-blur-sm">
                <ContactForm />
              </div>
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}