const express = require('express');
const cors = require('cors');
const path = require('path');
const fs = require('fs');
const fetch = require('node-fetch');
require('dotenv').config();

const app = express();
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
    doc.text = (doc.title || '') + '\n' + (doc.content || '');
    documents.push(doc);
  }
});

// Helper: get embedding from Groq
async function getEmbedding(text) {
  const response = await fetch('https://api.groq.com/openai/v1/embeddings', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${process.env.GROQ_API_KEY}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      input: text,
      model: 'text-embedding-3-small'
    })
  });

  const data = await response.json();
  return data.data[0].embedding;
}

// Compute cosine similarity
function cosineSimilarity(vecA, vecB) {
  const dot = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
  const magA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
  const magB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
  return dot / (magA * magB);
}

// Precompute embeddings for all documents
let documentEmbeddings = [];
(async () => {
  documentEmbeddings = await Promise.all(documents.map(doc => getEmbedding(doc.text)));
})();

app.post('/chat', async (req, res) => {
  const userMessage = req.body.message;

  try {
    const userEmbedding = await getEmbedding(userMessage);

    const scored = documents.map((doc, i) => ({
      doc,
      score: cosineSimilarity(userEmbedding, documentEmbeddings[i])
    }));

    scored.sort((a, b) => b.score - a.score);
    const topDocs = scored.slice(0, 3).map(d => d.doc.text).join('\n---\n');

    const systemPrompt = `You are a helpful assistant for Shiva Boys' Hindu College in Trinidad and Tobago. Use ONLY the following extracted information to answer:\n${topDocs}`;

    const chatResponse = await fetch('https://api.groq.com/openai/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${process.env.GROQ_API_KEY}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model: 'mixtral-8x7b-32768',
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: userMessage }
        ]
      })
    });

    const result = await chatResponse.json();
    const reply = result.choices[0]?.message?.content || "Sorry, I didn't understand that.";

    res.json({ response: reply });

  } catch (err) {
    console.error('Chat error:', err);
    res.status(500).json({ error: 'Failed to get response from Groq' });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
