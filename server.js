const express = require('express');
const cors = require('cors');
const path = require('path');
const fs = require('fs');
const { OpenAI } = require('openai');
require('dotenv').config();

const app = express();
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

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
    // Combine title and content for embedding
    doc.text = (doc.title || '') + '\n' + (doc.content || '');
    documents.push(doc);
  }
});

// Helper: get embedding for text
async function getEmbedding(text) {
  const embeddingResponse = await openai.embeddings.create({
    input: text,
    model: 'text-embedding-3-small', // or other embedding model
  });
  return embeddingResponse.data[0].embedding;
}

// Precompute embeddings for documents at startup
let documentEmbeddings = [];
(async () => {
  documentEmbeddings = await Promise.all(documents.map(doc => getEmbedding(doc.text)));
})();

// Compute cosine similarity between two vectors
function cosineSimilarity(vecA, vecB) {
  const dotProduct = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
  const magA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
  const magB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
  return dotProduct / (magA * magB);
}

app.post('/chat', async (req, res) => {
  const userMessage = req.body.message;

  try {
    // Get embedding for user query
    const userEmbedding = await getEmbedding(userMessage);

    // Find top 3 most similar docs
    const scoredDocs = documents.map((doc, i) => ({
      doc,
      score: cosineSimilarity(userEmbedding, documentEmbeddings[i])
    }));
    scoredDocs.sort((a, b) => b.score - a.score);
    const topDocs = scoredDocs.slice(0, 3).map(d => d.doc.text).join('\n---\n');

    // Send system prompt with retrieved context
    const messages = [
      {
        role: "system",
        content: `You are a helpful assistant for Shiva Boys' Hindu College in Trinidad and Tobago. Use ONLY the following extracted information from the website to answer user questions:\n${topDocs}`
      },
      { role: "user", content: userMessage }
    ];

    const completion = await openai.chat.completions.create({
      model: "gpt-4",
      messages: messages,
    });

    res.json({ response: completion.choices[0].message.content });
  } catch (err) {
    console.error('Error in /chat:', err);
    res.status(500).json({ error: 'Failed to get response from AI' });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
