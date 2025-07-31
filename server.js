const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
require('dotenv').config();

const app = express();
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

const RAG_ENABLED = process.env.RAG_ENABLED === 'true';

let documents = [];
let documentEmbeddings = [];

// Load website content from data/*.json files
function loadDocuments() {
  const dataDir = path.join(__dirname, process.env.DATA_PATH || 'data');
  const files = fs.readdirSync(dataDir);

  documents = files.flatMap(file => {
    const content = fs.readFileSync(path.join(dataDir, file), 'utf-8');
    try {
      const items = JSON.parse(content);
      return items.map(item => ({
        text: item.text,
        source: file
      }));
    } catch (err) {
      console.error(`Error parsing ${file}:`, err);
      return [];
    }
  });

  console.log(`Loaded ${documents.length} documents from website data.`);
}

// Get Groq embedding
async function getEmbedding(text) {
  try {
    const response = await fetch('https://api.groq.com/openai/v1/embeddings', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${process.env.GROQ_API_KEY}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        input: text,
        model: process.env.GROQ_MODEL || 'text-embedding-3-small'
      })
    });

    const data = await response.json();

    if (!data.data || !data.data[0]) {
      console.error('❌ Invalid embedding response:', JSON.stringify(data));
      return null;
    }

    return data.data[0].embedding;
  } catch (err) {
    console.error('❌ Error fetching embedding:', err);
    return null;
  }
}

// Dot product for similarity
function cosineSimilarity(a, b) {
  const dot = a.reduce((sum, val, i) => sum + val * b[i], 0);
  const normA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
  const normB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
  return dot / (normA * normB);
}

// Retrieve top 3 relevant documents
function getRelevantDocuments(queryEmbedding, topK = 3) {
  return documents
    .map((doc, i) => ({
      ...doc,
      similarity: cosineSimilarity(queryEmbedding, documentEmbeddings[i])
    }))
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, topK);
}

// Chat endpoint
app.post('/chat', async (req, res) => {
  const userMessage = req.body.message;
  let contextText = '';

  if (RAG_ENABLED) {
    const queryEmbedding = await getEmbedding(userMessage);
    if (queryEmbedding) {
      const topDocs = getRelevantDocuments(queryEmbedding);
      contextText = topDocs.map(doc => doc.text).join('\n\n');
    }
  }

  const payload = {
    model: 'mixtral-8x7b-32768',
    messages: [
      {
        role: 'system',
        content: `You are an assistant for Shiva Boys' Hindu College in Trinidad and Tobago. Answer using the provided website context if available.`
      },
      ...(RAG_ENABLED && contextText
        ? [{ role: 'system', content: `Website context:\n${contextText}` }]
        : []),
      {
        role: 'user',
        content: userMessage
      }
    ]
  };

  try {
    const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${process.env.GROQ_API_KEY}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload)
    });

    const data = await response.json();
    const reply = data.choices?.[0]?.message?.content;

    if (reply) {
      res.json({ response: reply });
    } else {
      console.error('Groq error:', JSON.stringify(data));
      res.status(500).json({ error: 'Failed to get response from Groq API' });
    }
  } catch (err) {
    console.error('Groq API call failed:', err);
    res.status(500).json({ error: 'Server error' });
  }
});

// Start server
const PORT = process.env.PORT || 10000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));

// Init: load and embed
(async () => {
  if (RAG_ENABLED) {
    loadDocuments();
    const embeddingResults = await Promise.all(documents.map(doc => getEmbedding(doc.text)));
    const dimension = 1536;
    documentEmbeddings = embeddingResults.map(e => e || Array(dimension).fill(0));
    console.log('Document embeddings ready.');
  }
})();
