require('dotenv').config();
const express = require('express');
const cors = require('cors');
const fetch = require('node-fetch');
const fs = require('fs');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 10000;
const GROQ_API_KEY = process.env.GROQ_API_KEY;
const GROQ_CHAT_MODEL = "llama3-8b-8192";
const EMBEDDING_MODEL = "nomic-embed-text-v1"; // supported embedding model

app.use(cors());
app.use(express.json());

const documents = [];
const embeddings = [];

// Load and embed documents on startup
async function loadDocuments() {
  const dataDir = path.join(__dirname, 'data');
  const files = fs.readdirSync(dataDir);

  for (const file of files) {
    const content = fs.readFileSync(path.join(dataDir, file), 'utf-8');
    const json = JSON.parse(content);
    for (const doc of json) {
      const embedding = await getEmbedding(doc.content);
      if (embedding) {
        documents.push(doc);
        embeddings.push(embedding);
      } else {
        console.error("❌ Failed to get embedding for:", doc.title);
      }
    }
  }
  console.log(`Loaded ${documents.length} documents from website data.`);
}

async function getEmbedding(text) {
  const response = await fetch('https://api.groq.com/openai/v1/embeddings', {
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

  const data = await response.json();

  if (!data.data || !data.data[0]) {
    console.error("❌ Invalid embedding response:", JSON.stringify(data));
    return null;
  }

  return data.data[0].embedding;
}

function cosineSimilarity(a, b) {
  const dot = a.reduce((sum, val, i) => sum + val * b[i], 0);
  const normA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
  const normB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
  return dot / (normA * normB);
}

async function getRelevantDocs(query) {
  const queryEmbedding = await getEmbedding(query);
  if (!queryEmbedding) return [];

  const similarities = embeddings.map((emb, i) => ({
    index: i,
    score: cosineSimilarity(queryEmbedding, emb)
  }));

  similarities.sort((a, b) => b.score - a.score);
  return similarities.slice(0, 5).map(sim => documents[sim.index]);
}

async function generateAnswer(userMessage, contextDocs) {
  const contextText = contextDocs.map(doc => `- ${doc.content}`).join('\n');

  const systemPrompt = `
You are the AI assistant for Shiva Boys' Hindu College. Answer user queries using both the provided context and your own knowledge, but always prioritize official school info if available.

Context:
${contextText}
`;

  const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${GROQ_API_KEY}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      model: GROQ_CHAT_MODEL,
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userMessage }
      ]
    })
  });

  const data = await response.json();

  if (data.choices && data.choices[0] && data.choices[0].message) {
    return data.choices[0].message.content.trim();
  } else {
    console.error("❌ Chat API error:", data);
    return "I'm sorry, I couldn't find an answer to your question.";
  }
}

app.post('/api/chat', async (req, res) => {
  const userMessage = req.body.message;

  if (!userMessage) {
    return res.status(400).json({ error: "Missing 'message' in request body." });
  }

  try {
    const relevantDocs = await getRelevantDocs(userMessage);
    const answer = await generateAnswer(userMessage, relevantDocs);
    res.json({ answer });
  } catch (err) {
    console.error("❌ Error handling request:", err);
    res.status(500).json({ error: "Internal Server Error" });
  }
});

loadDocuments().then(() => {
  console.log("Document embeddings ready.");
  app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
  });
});
