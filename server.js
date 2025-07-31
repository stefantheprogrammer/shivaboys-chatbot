import express from "express";
import fs from "fs";
import path from "path";
import dotenv from "dotenv";
import cors from "cors";
import { pipeline } from "@huggingface/inference";
import fetch from "node-fetch";
import { fileURLToPath } from "url";

dotenv.config();
const app = express();
app.use(express.json());
app.use(cors());

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const DATA_FOLDER = path.join(__dirname, "data");
let documents = [];
let documentEmbeddings = [];

// Load documents
function loadDocuments() {
  const files = fs.readdirSync(DATA_FOLDER).filter(f => f.endsWith(".json"));
  documents = [];
  for (const file of files) {
    const content = fs.readFileSync(path.join(DATA_FOLDER, file), "utf-8");
    try {
      const json = JSON.parse(content);
      json.forEach(entry => {
        if (entry.title && entry.content) {
          documents.push({ title: entry.title, content: entry.content });
        }
      });
    } catch (err) {
      console.error("Invalid JSON in", file);
    }
  }
  console.log(`Loaded ${documents.length} documents from website data.`);
}

// Compute local embeddings using HuggingFace
async function computeEmbeddings() {
  const embed = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
  documentEmbeddings = await Promise.all(documents.map(doc => embed(doc.content).then(v => v[0])));
  console.log("Document embeddings ready.");
}

// Cosine similarity
function cosineSimilarity(a, b) {
  const dot = a.reduce((sum, val, i) => sum + val * b[i], 0);
  const normA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
  const normB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
  return dot / (normA * normB);
}

// Get top matching document
async function getRelevantDocs(query) {
  const embed = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
  const queryVec = (await embed(query))[0];
  const scoredDocs = documents.map((doc, i) => ({
    ...doc,
    score: cosineSimilarity(queryVec, documentEmbeddings[i])
  }));
  return scoredDocs.sort((a, b) => b.score - a.score).slice(0, 3);
}

// Groq chat
async function chatWithGroq(systemPrompt, userPrompt) {
  const response = await fetch("https://api.groq.com/openai/v1/chat/completions", {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${process.env.GROQ_API_KEY}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      model: "llama3-70b-8192",
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userPrompt }
      ]
    })
  });
  const data = await response.json();
  return data.choices?.[0]?.message?.content || "Sorry, I couldn't respond at the moment.";
}

// API endpoint
app.post("/chat", async (req, res) => {
  const question = req.body.question;
  if (!question) return res.status(400).json({ error: "Missing question" });

  const relevantDocs = await getRelevantDocs(question);
  const context = relevantDocs.map(d => `${d.title}:
${d.content}`).join("

");
  const systemPrompt = `You are a helpful assistant for Shiva Boys' Hindu College. Use the following context from the website to answer the question:

${context}`;
  const answer = await chatWithGroq(systemPrompt, question);
  res.json({ answer });
});

// Init
app.listen(10000, async () => {
  console.log("Loading documents...");
  loadDocuments();
  console.log("Computing embeddings for documents...");
  await computeEmbeddings();
  console.log("Server running on port 10000");
});