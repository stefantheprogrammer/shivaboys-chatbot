require("dotenv").config();
const express = require("express");
const cors = require("cors");
const fs = require("fs");
const path = require("path");
const fetch = require("node-fetch");

const app = express();
app.use(cors());
app.use(express.json());

const PORT = process.env.PORT || 10000;
const GROQ_API_KEY = process.env.GROQ_API_KEY;
const GROQ_API_URL = "https://api.groq.com/openai/v1";

const documents = [];
const embeddings = [];

// Load all JSON files from the data folder
async function loadDocuments() {
  const dataDir = path.join(__dirname, "data");
  const files = fs.readdirSync(dataDir).filter(file => file.endsWith(".json"));

  for (const file of files) {
    const filePath = path.join(dataDir, file);
    const rawData = fs.readFileSync(filePath, "utf-8");
    const json = JSON.parse(rawData);

    for (const doc of json) {
      if (!doc.content || typeof doc.content !== "string" || doc.content.trim() === "") {
        console.error("⚠️ Skipped document with missing or invalid content:", doc.title || "No title");
        continue;
      }

      const embedding = await getEmbedding(doc.content);
      if (embedding) {
        documents.push(doc);
        embeddings.push(embedding);
      } else {
        console.error("❌ Failed to get embedding for:", doc.title || "Untitled document");
      }
    }
  }

  console.log(`Loaded ${documents.length} documents from website data.`);
}

// Groq API - get embeddings
async function getEmbedding(text) {
  try {
    const res = await fetch(`${GROQ_API_URL}/embeddings`, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${GROQ_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "gemma-7b-it", // Updated model
        input: text,
      }),
    });

    const json = await res.json();

    if (!json.data || !json.data[0]) {
      console.error("❌ Invalid embedding response:", JSON.stringify(json));
      return null;
    }

    return json.data[0].embedding;
  } catch (err) {
    console.error("❌ Embedding fetch error:", err.message);
    return null;
  }
}

// Groq API - chat completion
async function getGroqResponse(prompt) {
  const res = await fetch(`${GROQ_API_URL}/chat/completions`, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${GROQ_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: "mixtral-8x7b-32768",
      messages: [
        { role: "system", content: "You are a helpful assistant for a school. Use the documents provided as context but also provide natural AI reasoning if needed." },
        { role: "user", content: prompt }
      ],
      temperature: 0.7,
    }),
  });

  const json = await res.json();
  return json.choices?.[0]?.message?.content || "I'm not sure how to respond.";
}

// Cosine similarity
function cosineSimilarity(a, b) {
  const dotProduct = a.reduce((sum, val, i) => sum + val * b[i], 0);
  const magA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
  const magB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
  return dotProduct / (magA * magB);
}

// Endpoint for chat
app.post("/chat", async (req, res) => {
  const { prompt } = req.body;

  if (!prompt) return res.status(400).json({ error: "Missing prompt" });

  const promptEmbedding = await getEmbedding(prompt);
  if (!promptEmbedding) return res.status(500).json({ error: "Failed to get embedding for prompt" });

  // Get top 3 most similar documents
  const scores = embeddings.map(e => cosineSimilarity(promptEmbedding, e));
  const topIndexes = scores
    .map((score, idx) => ({ score, idx }))
    .sort((a, b) => b.score - a.score)
    .slice(0, 3)
    .map(obj => obj.idx);

  const context = topIndexes.map(i => documents[i].content).join("\n\n");

  const finalPrompt = `Use the following website information to help answer the question:\n\n${context}\n\nQuestion: ${prompt}`;
  const answer = await getGroqResponse(finalPrompt);

  res.json({ answer });
});

// Start server
loadDocuments().then(() => {
  console.log("Document embeddings ready.");
  app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
  });
});
