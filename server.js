const express = require('express');
const cors = require('cors');
const path = require('path');
const fs = require('fs');
require('dotenv').config();

const { GroqClient } = require('@groq/client');

const app = express();

const client = new GroqClient({
  apiKey: process.env.GROQ_API_KEY,
  model: process.env.GROQ_MODEL || 'mixtral-8x7b-32768',
});

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// Load all JSON docs from data folder at startup
const DATA_PATH = process.env.DATA_PATH || 'data/';
const documents = [];

fs.readdirSync(DATA_PATH).forEach(file => {
  if (file.endsWith('.json')) {
    const raw = fs.readFileSync(path.join(DATA_PATH, file));
    const doc = JSON.parse(raw);
    // Combine title and content for retrieval
    doc.text = (doc.title || '') + '\n' + (doc.content || '');
    documents.push(doc);
  }
});

// Basic cosine similarity helper function for embeddings (if you want to embed locally)
function cosineSimilarity(vecA, vecB) {
  const dotProduct = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
  const magA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
  const magB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
  return dotProduct / (magA * magB);
}

app.post('/chat', async (req, res) => {
  const userMessage = req.body.message;

  try {
    // --- Retrieval Augmentation using Groq ---

    // Option 1: Use Groq's built-in RAG endpoint (if available)
    // e.g. client.complete() with retrieval documents

    // Option 2: If no embedding endpoint, do simple text matching here (or precompute embeddings offline)
    // For demo: concatenate all docs (or top N) and send as context prompt

    const contextText = documents.map(d => d.text).join('\n---\n');

    const prompt = `You are a helpful assistant for Shiva Boys' Hindu College in Trinidad and Tobago.
Use ONLY the following extracted information from the website to answer user questions:
${contextText}

User question: ${userMessage}
Answer:`;

    // Call Groq completion endpoint
    const completion = await client.complete({
      prompt,
      maxTokens: 500,
      temperature: 0.2,
    });

    res.json({ response: completion.text.trim() });

  } catch (err) {
    console.error('Error in /chat:', err);
    res.status(500).json({ error: 'Failed to get response from AI' });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
