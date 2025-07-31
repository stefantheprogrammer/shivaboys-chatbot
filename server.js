const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const fetch = require('node-fetch');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 10000;
const DATA_PATH = path.join(__dirname, 'data');

if (!process.env.GROQ_API_KEY) {
  console.error('ERROR: GROQ_API_KEY not set in environment.');
  process.exit(1);
}

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// In-memory docs array
const documents = [];

// Load JSON documents from data folder
function loadDocuments() {
  console.log('Loading documents...');
  documents.length = 0; // clear existing

  const files = fs.readdirSync(DATA_PATH);
  files.forEach((file) => {
    if (file.endsWith('.json')) {
      try {
        const raw = fs.readFileSync(path.join(DATA_PATH, file), 'utf-8');
        const items = JSON.parse(raw);
        if (!Array.isArray(items)) {
          console.warn(`Skipped ${file}: JSON root is not an array`);
          return;
        }
        items.forEach((item, idx) => {
          if (item.title && item.content) {
            documents.push({
              title: item.title,
              content: item.content,
              text: item.title + '\n' + item.content,
            });
          } else {
            console.warn(`Skipped item ${idx} in ${file}: Missing title or content`);
          }
        });
      } catch (e) {
        console.error(`Error parsing ${file}:`, e);
      }
    }
  });

  console.log(`Loaded ${documents.length} documents from website data.`);
}

loadDocuments();

// Groq API info
const GROQ_API_URL = 'https://api.groq.com/v1/embeddings';
const GROQ_CHAT_URL = 'https://api.groq.com/v1/chat/completions';

// Get embedding for a text using Groq embeddings API
async function getEmbedding(text) {
  if (!text) {
    console.warn('getEmbedding called with empty text');
    return null;
  }
  try {
    const body = {
      model: 'gemma-13b', // Use latest recommended model from Groq docs
      input: text,
    };

    const res = await fetch(GROQ_API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${process.env.GROQ_API_KEY}`,
      },
      body: JSON.stringify(body),
    });
    const data = await res.json();

    if (data.error) {
      console.error('Embedding API error:', data.error);
      return null;
    }

    if (!data.data || !data.data[0] || !data.data[0].embedding) {
      console.error('Invalid embedding response:', data);
      return null;
    }

    return data.data[0].embedding;
  } catch (err) {
    console.error('Failed to fetch embedding:', err);
    return null;
  }
}

// Compute cosine similarity between two vectors
function cosineSimilarity(a, b) {
  const dot = a.reduce((sum, x, i) => sum + x * b[i], 0);
  const magA = Math.sqrt(a.reduce((sum, x) => sum + x * x, 0));
  const magB = Math.sqrt(b.reduce((sum, x) => sum + x * x, 0));
  if (magA === 0 || magB === 0) return 0;
  return dot / (magA * magB);
}

let documentEmbeddings = [];

// Precompute embeddings for documents at startup
async function prepareDocumentEmbeddings() {
  console.log('Computing embeddings for documents...');
  documentEmbeddings = [];

  for (const doc of documents) {
    const emb = await getEmbedding(doc.text);
    if (emb) {
      documentEmbeddings.push(emb);
    } else {
      // For failed embeddings, add zero vector of same length (assuming 1536 dims)
      documentEmbeddings.push(Array(1536).fill(0));
      console.warn(`Failed embedding for document titled: "${doc.title}"`);
    }
  }

  console.log('Document embeddings ready.');
}

(async () => {
  await prepareDocumentEmbeddings();
})();

// Chat completion using Groq chat API
async function getChatCompletion(messages) {
  try {
    const body = {
      model: 'gemma-13b-chat',
      messages,
    };

    const res = await fetch(GROQ_CHAT_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${process.env.GROQ_API_KEY}`,
      },
      body: JSON.stringify(body),
    });

    const data = await res.json();

    if (data.error) {
      console.error('Chat API error:', data.error);
      return null;
    }

    if (!data.choices || !data.choices[0] || !data.choices[0].message) {
      console.error('Invalid chat response:', data);
      return null;
    }

    return data.choices[0].message.content;
  } catch (err) {
    console.error('Failed to get chat completion:', err);
    return null;
  }
}

app.post('/chat', async (req, res) => {
  const userMessage = req.body.message;
  if (!userMessage) {
    return res.status(400).json({ error: 'Missing message in request' });
  }

  try {
    // Get user query embedding
    const userEmbedding = await getEmbedding(userMessage);
    if (!userEmbedding) {
      return res.status(500).json({ error: 'Failed to get embedding for user message' });
    }

    // Find top 3 documents most similar to user query
    const scoredDocs = documents
      .map((doc, idx) => ({
        doc,
        score: cosineSimilarity(userEmbedding, documentEmbeddings[idx]),
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 3);

    // Combine top docs as context string
    const contextText = scoredDocs.map(d => d.doc.text).join('\n---\n');

    // Prepare messages array for chat completion
    const messages = [
      {
        role: 'system',
        content: `You are a helpful assistant for Shiva Boys' Hindu College in Trinidad and Tobago. Use ONLY the following extracted information from the website to answer user questions:\n${contextText}`,
      },
      {
        role: 'user',
        content: userMessage,
      },
    ];

    // Get response from Groq chat API
    const aiResponse = await getChatCompletion(messages);

    if (!aiResponse) {
      return res.status(500).json({ error: 'Failed to get AI response' });
    }

    res.json({ response: aiResponse });
  } catch (err) {
    console.error('Error in /chat:', err);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Start server
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});