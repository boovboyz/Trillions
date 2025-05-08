'use client'; // For potential client-side interactions in the future

// No longer using next/link for same-page scrolling
// import Link from 'next/link';

export default function Navbar() {
  // Basic smooth scroll handler (can be enhanced)
  const handleScroll = (e: React.MouseEvent<HTMLAnchorElement>, id: string) => {
    e.preventDefault();
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <nav className="bg-gradient-to-r from-gray-900 via-gray-800 to-black text-white p-4 shadow-lg sticky top-0 z-50 border-b border-gray-700/50">
      <div className="container mx-auto flex justify-between items-center">
        {/* Use an anchor tag for the logo to scroll to top or stay on page */}
        <a href="#hero" onClick={(e) => handleScroll(e, 'hero')} className="flex items-center space-x-2 group">
          <svg className="w-8 h-8 text-blue-400 group-hover:text-teal-300 transition-colors duration-300" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg> 
          <span className="text-2xl font-bold tracking-tight group-hover:text-gray-100 transition-colors duration-300">
            ARTHA AI
          </span>
        </a>
        
        {/* Navigation Links - Pointing to section IDs */}
        <div className="hidden md:flex items-center space-x-8">
          <a href="#features" onClick={(e) => handleScroll(e, 'features')} className="text-base font-medium hover:text-teal-300 transition-colors duration-300">
            Features
          </a>
          <a href="#how-it-works" onClick={(e) => handleScroll(e, 'how-it-works')} className="text-base font-medium hover:text-teal-300 transition-colors duration-300">
            How It Works
          </a>
          <a href="#demo" onClick={(e) => handleScroll(e, 'demo')} className="text-base font-medium hover:text-teal-300 transition-colors duration-300">
            Live Demo
          </a>
          <a href="#contact" onClick={(e) => handleScroll(e, 'contact')} className="text-base font-medium hover:text-teal-300 transition-colors duration-300">
            Contact
          </a>
        </div>
        
        {/* Mobile Menu Button Placeholder */}
        <div className="md:hidden">
          <button className="p-2 rounded-md hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-white">
             <svg className="h-6 w-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6h16M4 12h16M4 18h16"></path></svg>
          </button>
          {/* Mobile menu itself would be implemented here, toggled by state */}
        </div>
      </div>
    </nav>
  );
} 