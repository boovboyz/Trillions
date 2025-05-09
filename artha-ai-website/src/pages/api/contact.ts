import type { NextApiRequest, NextApiResponse } from 'next';
import { createClient } from '@supabase/supabase-js';

// Initialize Supabase client with environment variables
// Ensure these are set in your .env.local file
const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseServiceRoleKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

if (!supabaseUrl || !supabaseServiceRoleKey) {
  console.error("Supabase URL or Service Role Key is missing. Check .env.local file.");
  // You might want to throw an error here or handle it appropriately
  // For now, let's log it and allow the server to start, but API will fail.
}

// It's good practice to create the client only once if possible,
// but for serverless functions, creating it per request is also common.
// If you have a central place for initializing Supabase, use that.
// For this example, we initialize it here.
const supabaseAdmin = supabaseUrl && supabaseServiceRoleKey ? createClient(supabaseUrl, supabaseServiceRoleKey) : null;

type Data = {
  message?: string;
  error?: string;
};

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse<Data>
) {
  if (!supabaseAdmin) {
    return res.status(500).json({ error: 'Supabase client not initialized. Server configuration error.' });
  }

  if (req.method === 'POST') {
    const { name, email, message } = req.body;

    if (!name || !email || !message) {
      return res.status(400).json({ error: 'Missing required fields: name, email, and message are required.' });
    }

    try {
      const { data, error } = await supabaseAdmin
        .from('contact_submissions')
        .insert([{ name, email, message }])
        .select(); // .select() is optional, but can be useful to confirm the insert

      if (error) {
        console.error('Supabase error:', error);
        return res.status(500).json({ error: error.message || 'Failed to submit message to Supabase.' });
      }

      // Successfully inserted
      return res.status(200).json({ message: 'Message sent successfully!' });

    } catch (error: any) {
      console.error('Handler error:', error);
      return res.status(500).json({ error: error.message || 'An unexpected error occurred.' });
    }
  } else {
    // Handle any other HTTP method
    res.setHeader('Allow', ['POST']);
    res.status(405).json({ error: `Method ${req.method} Not Allowed` });
  }
} 