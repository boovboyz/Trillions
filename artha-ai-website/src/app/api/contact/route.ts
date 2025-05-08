import { NextResponse } from 'next/server';

export async function POST(request: Request) {
  console.log('API route /api/contact called');
  try {
    const body = await request.json();
    const { name, email, message } = body;

    // Basic validation
    if (!name || !email || !message) {
      return NextResponse.json({ error: 'Missing required fields' }, { status: 400 });
    }

    // --- Placeholder for actual processing --- 
    // In a real application, you would:
    // 1. Validate the data more thoroughly (e.g., email format).
    // 2. Send an email using a service like Resend, SendGrid, or Nodemailer.
    // 3. Or, save the message to your database (e.g., a 'contacts' table in Supabase).
    
    console.log('Received contact form submission:');
    console.log(`Name: ${name}`);
    console.log(`Email: ${email}`);
    console.log(`Message: ${message}`);
    // Simulate processing time
    await new Promise(resolve => setTimeout(resolve, 500)); 
    // --- End Placeholder --- 

    // Return success response
    return NextResponse.json({ message: 'Message received successfully!' }, { status: 200 });

  } catch (error: any) {
    console.error('Error processing contact form:', error);
    return NextResponse.json(
      { error: 'Internal server error processing message.', details: error.message || error.toString() }, 
      { status: 500 }
    );
  }
} 