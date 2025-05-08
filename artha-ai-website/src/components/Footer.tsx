'use client'; // Good practice, though not strictly needed for this static footer yet

export default function Footer() {
  return (
    <footer className="bg-gray-100 text-gray-700 p-6 text-center mt-auto border-t">
      <div className="container mx-auto">
        <p className="mb-2">&copy; {new Date().getFullYear()} ARTHA AI. All rights reserved.</p>
        <p className="text-xs text-gray-500">
          Disclaimer: Trading involves substantial risk of loss and is not suitable for every investor. Past performance is not indicative of future results. ARTHA AI provides analytical tools and does not offer financial advice.
        </p>
      </div>
    </footer>
  );
} 