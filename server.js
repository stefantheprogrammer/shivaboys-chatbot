const express = require('express');
const cors = require('cors');
const path = require('path');
const fs = require('fs');
const fetch = require('node-fetch');
require('dotenv').config();

const app = express();

const GROQ_API_KEY = process.env.GROQ_API_KEY;
const GROQ_API_URL = 'https://api.groq.com/v1/embeddings';

// Use the updated Groq embedding model
const EMBEDDING_MODEL = 'gemma2-9b-it';

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

const DATA_PATH = process.env.DATA_PATH || 'data/';
const documents = [];

// Load documents from JSON files
function loadDocuments() {
  const docs = [];
  const files = fs.readdirSync(DATA_PATH);

  files.forEach(file => {
    if (file.endsWith('.json')) {
      try {
        const raw = fs.readFileSync(path.join(DATA_PATH, file), 'utf-8');
        const jsonData = JSON.parse(raw);

        // Accept array of objects with 'text' or single object with 'title' and 'content'
        if (Array.isArray(jsonData)) {
          jsonData.forEach(item => {
            if (item.text && item.text.trim().length > 0) {
              docs.push({
                title: item.title || '',
                content: item.text,
                text: (item.title || '') + '\n' + item.text
              });
            }
          });
        } else if (jsonData.title && jsonData.content) {
          docs.push({
            title: jsonData.title,
            content: jsonData.content,
            text: jsonData.title + '\n' + jsonData.content
          });
        } else {
          console.warn(`Skipping invalid document structure in ${file}`);
        }
      } catch (err) {
        console.error(`Error parsing ${file}:`, err);
      }
    }
  });

  return docs;
}

async function getEmbedding(text) {
  if (!text) {
    console.error('Empty text passed to getEmbedding');
    return null;
  }

  try {
    const res = await fetch(GROQ_API_URL, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${GROQ_API_KEY}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model: EMBEDDING_MODEL,
        input: text
      })
    });

    const data = await res.json();

    if (!data.data || !data.data[0] || !data.data[0].embedding) {
      console.error('❌ Invalid embedding response:', JSON.stringify(data));
      return null;
    }

    return data.data[0].embedding;
  } catch (error) {
    console.error('Error fetching embedding:', error);
    return null;
  }
}

// Compute cosine similarity between two vectors
function cosineSimilarity(vecA, vecB) {
  if (!vecA || !vecB || vecA.length !== vecB.length) return 0;
  let dotProduct = 0,
    magA = 0,
    magB = 0;

  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    magA += vecA[i] * vecA[i];
    magB += vecB[i] * vecB[i];
  }

  if (magA === 0 || magB === 0) return 0;
  return dotProduct / (Math.sqrt(magA) * Math.sqrt(magB));
}

// Load documents and precompute embeddings at startup
let documentEmbeddings = [];

(async () => {
  console.log('Loading documents...');
  documents.push(...loadDocuments());
  console.log(`Loaded ${documents.length} documents from website data.`);

  console.log('Computing embeddings...');
  const embeddings = await Promise.all(
    documents.map(async (doc) => {
      const emb = await getEmbedding(doc.text);
      if (!emb) {
        console.warn(`Failed to get embedding for document: ${doc.title || 'No title'}`);
      }
      return emb;
    })
  );
  // Filter out documents with failed embeddings
  documentEmbeddings = embeddings;
  console.log('Document embeddings ready.');
})();

app.post('/chat', async (req, res) => {
  const userMessage = req.body.message;

  if (!userMessage) {
    return res.status(400).json({ error: 'No message provided' });
  }

  try {
    const userEmbedding = await getEmbedding(userMessage);
    if (!userEmbedding) {
      return res.status(500).json({ error: 'Failed to get embedding for user message' });
    }

    // Compute similarity scores
    const scoredDocs = documents
      .map((doc, i) => ({
        doc,
        score: cosineSimilarity(userEmbedding, documentEmbeddings[i])
      }))
      .filter(item => item.score && item.doc) // remove invalids
      .sort((a, b) => b.score - a.score);

    // Top 3 relevant docs
    const topDocs = scoredDocs.slice(0, 3).map(d => d.doc.text).join('\n---\n');

    // Compose system prompt with retrieved context and fallback instruction
    const messages = [
      {
        role: 'system',
        content: `You are a helpful assistant for Shiva Boys' Hindu College in Trinidad and Tobago. Use ONLY the following extracted information from the website to answer user questions:\n${topDocs}\nIf the answer is not contained within the above information, you may answer from your general knowledge.`
      },
      { role: 'user', content: userMessage }
    ];

    // Call Groq chat completions endpoint (replace with your chat API if different)
    // NOTE: Adjust below to your actual chat endpoint or use OpenAI if preferred

    const chatResponse = await fetch('https://api.groq.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${GROQ_API_KEY}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model: 'gemma2-9b-it',
        messages: messages
      })
    });

    const chatData = await chatResponse.json();

    if (!chatData.choices || !chatData.choices[0] || !chatData.choices[0].message) {
      return res.status(500).json({ error: 'Invalid response from chat API' });
    }

    res.json({ response: chatData.choices[0].message.content });
  } catch (err) {
    console.error('Error in /chat:', err);
    res.status(500).json({ error: 'Failed to get response from AI' });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
