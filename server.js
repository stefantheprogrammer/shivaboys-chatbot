const express = require('express');
const cors = require('cors');
const path = require('path');
require('dotenv').config();
const { OpenAI } = require('openai'); // For OpenAI or Groq

const app = express();
const port = process.env.PORT || 3000;

// Middlewares
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// Serve widget.html with proper CORS headers
app.get('/widget.html', (req, res) => {
  res.setHeader('Content-Type', 'text/html');
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.sendFile(path.join(__dirname, 'public', 'widget.html'));
});

// Serve widget.js with proper headers (optional)
app.get('/widget.js', (req, res) => {
  res.setHeader('Content-Type', 'application/javascript');
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.sendFile(path.join(__dirname, 'public', 'widget.js'));
});

// Chat endpoint
app.post('/chat', async (req, res) => {
  const userMessage = req.body.message;

  if (!userMessage) {
    return res.status(400).json({ error: 'Missing user message' });
  }

  try {
    const openai = new OpenAI({
      apiKey: process.env.GROQ_API_KEY || process.env.OPENAI_API_KEY,
      baseURL: process.env.GROQ_API_KEY
        ? 'https://api.groq.com/openai/v1'
        : undefined,
    });

    const completion = await openai.chat.completions.create({
      model: process.env.GROQ_API_KEY ? 'llama3-8b-8192' : 'gpt-3.5-turbo',
      messages: [
        { role: 'system', content: 'You are an AI assistant for a secondary school in Trinidad and Tobago. Answer helpfully, clearly, and briefly.' },
        { role: 'user', content: userMessage },
      ],
    });

    const reply = completion.choices[0]?.message?.content;
    res.json({ reply: reply || "I'm not sure how to respond to that." });

  } catch (error) {
    console.error('OpenAI API error:', error);
    res.status(500).json({ error: 'An error occurred while processing your request.' });
  }
});

// Start the server
app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});
