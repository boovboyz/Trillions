'use client'; // Good practice, though not strictly needed for this static footer yet

import React from 'react';
import { Instagram, Linkedin, Twitter } from 'lucide-react';
import Image from 'next/image';

export default function Footer() {
  return (
    <footer className="bg-gradient-to-r from-gray-900 via-gray-800 to-black text-white py-10">
      <div className="container mx-auto px-4 md:px-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-12">
          <div className="space-y-4">
            <div className="flex items-center space-x-2">
              <div className="relative h-10 w-10 flex items-center justify-center">
                <Image src="/artha_logo.png" alt="VectorQuant AI Logo" width={40} height={40} className="object-contain" />
              </div>
              <span className="text-xl font-bold">VectorQuant AI</span>
            </div>
            <p className="text-gray-300 text-sm">
              Revolutionizing finance through the thoughtful application of artificial intelligence and algorithmic trading.
            </p>
          </div>

          <div className="space-y-4">
            <h3 className="text-lg font-semibold">Quick Links</h3>
            <nav className="flex flex-col space-y-2">
              <a href="#features" className="text-gray-300 hover:text-white transition-colors duration-200">Features</a>
              <a href="#how-it-works" className="text-gray-300 hover:text-white transition-colors duration-200">How It Works</a>
              <a href="#demo" className="text-gray-300 hover:text-white transition-colors duration-200">Live Demo</a>
              <a href="#contact" className="text-gray-300 hover:text-white transition-colors duration-200">Contact</a>
            </nav>
          </div>

          <div className="space-y-4">
            <h3 className="text-lg font-semibold">Connect With Us</h3>
            <div className="flex space-x-4">
              <a href="https://x.com/VectorQuantAI" target="_blank" rel="noopener noreferrer" className="text-gray-300 hover:text-white transition-colors duration-200">
                <Twitter className="h-5 w-5" />
              </a>
              <a href="https://www.linkedin.com/in/vectorquant-ai-0b30a4365" target="_blank" rel="noopener noreferrer" className="text-gray-300 hover:text-white transition-colors duration-200">
                <Linkedin className="h-5 w-5" />
              </a>
              <a href="https://www.instagram.com/vectorquantai/" target="_blank" rel="noopener noreferrer" className="text-gray-300 hover:text-white transition-colors duration-200">
                <Instagram className="h-5 w-5" />
              </a>
            </div>
            <p className="text-gray-400 text-sm">admin@vectorquantai.com</p>
          </div>
        </div>

        <div className="border-t border-gray-700 mt-10 pt-6 flex flex-col md:flex-row justify-between items-center">
          <p className="text-gray-400 text-sm">&copy; {new Date().getFullYear()} VectorQuant AI. All rights reserved.</p>
          <p className="text-xs text-gray-500 mt-2 md:mt-0 max-w-2xl text-center md:text-right">
            Disclaimer: Trading involves substantial risk of loss and is not suitable for every investor. Past performance is not indicative of future results. VectorQuant AI provides analytical tools and does not offer financial advice.
          </p>
        </div>
      </div>
    </footer>
  );
} 