import express from "express";
import cors from "cors";
import morgan from "morgan";
import dotenv from "dotenv";
import axios from "axios";
import fs from "fs/promises";
import path from "path";

dotenv.config();
const app = express();
const port = process.env.PORT || 3000;

app.use(cors());
app.use(morgan("dev"));
app.use(express.json());

const hfApi = axios.create({
  baseURL: "https://api-inference.huggingface.co",
  headers: {
    Authorization: `Bearer ${process.env.HUGGINGFACE_API_KEY}`,
    "Content-Type": "application/json"
  }
});

// Load local `.json` files from `/data` directory
async function loadDocuments() {
  const files = await fs.readdir(path.join(__dirname, "data"));
  const docs = [];
  for (const f of files) {
    const content = await fs.readFile(path.join(__dirname, "data", f), "utf8");
    const obj = JSON.parse(content);
    docs.push(`${obj.title || ""}\n${obj.content || ""}`);
  }
  return docs;
}

let documents = [];
let documentEmbeddings = [];

async function computeEmbeddings() {
  const docs = await loadDocuments();
  documents = docs;

  const resp = await hfApi.post("/embeddings", {
    model: "sentence-transformers/all-MiniLM-L6-v2",
    inputs: docs
  });

  documentEmbeddings = resp.data;
}

computeEmbeddings().catch(console.error);

function cosine(a, b) {
  let dot = 0, magA = 0, magB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    magA += a[i] * a[i];
    magB += b[i] * b[i];
  }
  return dot / (Math.sqrt(magA) * Math.sqrt(magB));
}

app.post("/chat", async (req, res) => {
  const { question } = req.body;
  if (!question) return res.status(400).json({ error: "Question is required" });

  // Create embedding for question
  const qEmbedRes = await hfApi.post("/embeddings", {
    model: "sentence-transformers/all-MiniLM-L6-v2",
    inputs: [question]
  });
  const qEmbed = qEmbedRes.data[0];

  // Find top-3 docs
  const scores = documentEmbeddings.map((de, i) => ({ idx: i, score: cosine(qEmbed, de) }));
  const top = scores.sort((a, b) => b.score - a.score).slice(0, 3);
  const context = top.map(o => documents[o.idx]).join("\n---\n");

  // Ask chat
  const prompt = `Use the context below to answer:\n${context}\n\nQuestion: ${question}`;

  const chatResponse = await hfApi.post("/text-generation", {
    model: "mistralai/Mixtral-8x7B-Instruct-v0.1",
    inputs: prompt,
    parameters: { max_new_tokens: 200, temperature: 0.3 }
  });

  const answer = chatResponse.data[0]?.generated_text || chatResponse.data.generated_text;
  res.json({ answer });
});

app.listen(port, () => console.log(`Server listening on ${port}`));
