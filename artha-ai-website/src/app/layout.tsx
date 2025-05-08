import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Navbar from "@/components/Navbar"; // Assuming @ is configured for src
import Footer from "@/components/Footer"; // Or use relative paths like ../components/

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "ARTHA AI - Intelligent Trading",
  description: "Maximize profits and minimize risk with AI-powered trading.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${inter.className} flex flex-col min-h-screen bg-gray-50 text-gray-900`}>
        <Navbar />
        <main className="flex-grow container mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {children}
        </main>
        <Footer />
      </body>
    </html>
  );
}
