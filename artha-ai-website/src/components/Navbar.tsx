'use client'; // For potential client-side interactions in the future

import React, { useState, useEffect } from 'react';
import { Menu, X } from 'lucide-react';
import Image from 'next/image';

// No longer using next/link for same-page scrolling
// import Link from 'next/link';

export default function Navbar() {
  const [isScrolled, setIsScrolled] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  // Update scroll state
  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 10);
    };
    
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // Smooth scroll handler
  const handleScroll = (e: React.MouseEvent<HTMLAnchorElement>, id: string) => {
    e.preventDefault();
    setIsMobileMenuOpen(false);
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <nav 
      className={`fixed top-0 w-full z-50 transition-all duration-300 ${
        isScrolled 
          ? 'bg-white/90 text-gray-900 shadow-md backdrop-blur-md dark:bg-gray-900/90 dark:text-white' 
          : 'bg-transparent text-white'
      }`}
    >
      <div className="container mx-auto px-4 md:px-6">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <a href="#hero" onClick={(e) => handleScroll(e, 'hero')} 
             className="flex items-center space-x-2 group">
            <div className="relative h-12 w-12 flex items-center justify-center">
              <Image src="/artha_logo.png" alt="Artha AI Logo" width={48} height={48} className="object-contain" />
            </div>
            <span className={`text-xl font-bold tracking-tight font-heading ${
              isScrolled ? 'text-gray-900 dark:text-white' : 'text-white'
            }`}>
              VectorQuant AI
            </span>
          </a>
          
          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center space-x-1">
            <NavLink href="#features" id="features" onClick={handleScroll} isScrolled={isScrolled}>Features</NavLink>
            <NavLink href="#how-it-works" id="how-it-works" onClick={handleScroll} isScrolled={isScrolled}>How It Works</NavLink>
            <NavLink href="#demo" id="demo" onClick={handleScroll} isScrolled={isScrolled}>Live Demo</NavLink>
            <NavLink href="#contact" id="contact" onClick={handleScroll} isScrolled={isScrolled} 
                     className="ml-2 px-4 py-2 rounded-full bg-gradient-to-r from-blue-600 to-teal-500 text-white hover:from-blue-700 hover:to-teal-600">
              Contact Us
            </NavLink>
          </div>
          
          {/* Mobile Menu Button */}
          <div className="md:hidden">
            <button 
              onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
              className={`p-2 rounded-md focus:outline-none focus:ring-2 focus:ring-offset-2 ${
                isScrolled 
                  ? 'hover:bg-gray-200 focus:ring-blue-500 text-gray-900 dark:text-white dark:hover:bg-gray-800' 
                  : 'hover:bg-white/10 focus:ring-white text-white'
              }`}
            >
              {isMobileMenuOpen ? (
                <X className="h-6 w-6" />
              ) : (
                <Menu className="h-6 w-6" />
              )}
            </button>
          </div>
        </div>
        
        {/* Mobile Menu */}
        {isMobileMenuOpen && (
          <div className="md:hidden">
            <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3 bg-white dark:bg-gray-900 rounded-lg shadow-lg mt-2">
              <MobileNavLink href="#features" id="features" onClick={handleScroll}>Features</MobileNavLink>
              <MobileNavLink href="#how-it-works" id="how-it-works" onClick={handleScroll}>How It Works</MobileNavLink>
              <MobileNavLink href="#demo" id="demo" onClick={handleScroll}>Live Demo</MobileNavLink>
              <MobileNavLink href="#contact" id="contact" onClick={handleScroll} className="bg-gradient-to-r from-blue-600 to-teal-500 text-white">
                Contact Us
              </MobileNavLink>
            </div>
          </div>
        )}
      </div>
    </nav>
  );
}

// Desktop nav link component
function NavLink({ 
  href, 
  id, 
  onClick, 
  isScrolled, 
  className = '', 
  children 
}: { 
  href: string;
  id: string;
  onClick: (e: React.MouseEvent<HTMLAnchorElement>, id: string) => void;
  isScrolled: boolean;
  className?: string;
  children: React.ReactNode;
}) {
  return (
    <a 
      href={href} 
      onClick={(e: React.MouseEvent<HTMLAnchorElement>) => onClick(e, id)}
      className={`px-3 py-2 rounded-md text-sm font-medium transition-colors duration-200 ${className} ${
        isScrolled 
          ? 'hover:bg-gray-100 hover:text-blue-600 dark:hover:bg-gray-800 dark:hover:text-blue-400' 
          : 'hover:bg-white/10 hover:text-white'
      }`}
    >
      {children}
    </a>
  );
}

// Mobile nav link component
function MobileNavLink({ 
  href, 
  id, 
  onClick, 
  className = '', 
  children 
}: { 
  href: string;
  id: string;
  onClick: (e: React.MouseEvent<HTMLAnchorElement>, id: string) => void;
  className?: string;
  children: React.ReactNode;
}) {
  return (
    <a 
      href={href} 
      onClick={(e: React.MouseEvent<HTMLAnchorElement>) => onClick(e, id)}
      className={`block px-3 py-2 rounded-md text-base font-medium text-gray-900 dark:text-white hover:bg-gray-100 dark:hover:bg-gray-800 ${className}`}
    >
      {children}
    </a>
  );
} 