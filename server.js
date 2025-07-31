const express = require("express");
const fs = require("fs");
const path = require("path");
const cors = require("cors");
const { Groq } = require("groq-sdk");

const app = express();
const port = process.env.PORT || 10000;
const groq = new Groq({ apiKey: process.env.GROQ_API_KEY || "your-groq-api-key" });

app.use(cors());
app.use(express.json());

// Helper: Load and flatten all documents
function loadDocuments(folderPath) {
  const files = fs.readdirSync(folderPath);
  return files.flatMap((file) => {
    const filePath = path.join(folderPath, file);
    const content = fs.readFileSync(filePath, "utf8");
    try {
      const json = JSON.parse(content);
      const items = Array.isArray(json) ? json : json.items;
      return items.map((item) => ({
        content: item.text || item.content || JSON.stringify(item),
        source: file,
      }));
    } catch (error) {
      console.error(`Error parsing ${file}: ${error}`);
      return [];
    }
  });
}

// Use Nomic embeddings instead of decommissioned Mixtral
async function getEmbedding(text) {
  const res = await groq.embeddings.create({
    model: "nomic-embed-text-v1",
    input: text,
  });

  if (!res.data || !Array.isArray(res.data) || !res.data[0].embedding) {
    console.error("❌ Invalid embedding response:", JSON.stringify(res));
    return null;
  }

  return res.data[0].embedding;
}

// Load docs and embed them
let embeddedDocs = [];
async function initializeEmbeddings() {
  const docs = loadDocuments(path.join(__dirname, "data"));
  const embeddings = await Promise.all(
    docs.map(async (doc) => {
      const embedding = await getEmbedding(doc.content);
      return embedding ? { ...doc, embedding } : null;
    })
  );
  embeddedDocs = embeddings.filter(Boolean);
  console.log("Document embeddings ready.");
}

// Cosine similarity
function cosineSimilarity(a, b) {
  const dot = a.reduce((sum, val, i) => sum + val * b[i], 0);
  const magA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
  const magB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
  return dot / (magA * magB);
}

// Chat endpoint
app.post("/chat", async (req, res) => {
  const { question } = req.body;
  if (!question) return res.status(400).send({ error: "Missing question" });

  const userEmbedding = await getEmbedding(question);
  if (!userEmbedding) return res.status(500).send({ error: "Embedding failed" });

  const ranked = embeddedDocs
    .map((doc) => ({
      ...doc,
      score: cosineSimilarity(userEmbedding, doc.embedding),
    }))
    .sort((a, b) => b.score - a.score);

  const topDocs = ranked.slice(0, 5);
  const context = topDocs.map((d) => d.content).join("\n---\n");

  const prompt = `Use the context below to answer the user's question.\n\nContext:\n${context}\n\nQuestion: ${question}\nAnswer:`;

  const completion = await groq.chat.completions.create({
    messages: [{ role: "user", content: prompt }],
    model: "mixtral-8x7b-32768", // or replace if using a different chat model
  });

  const reply = completion.choices?.[0]?.message?.content;
  res.send({ answer: reply || "No response" });
});

initializeEmbeddings().then(() => {
  console.log("Loaded", embeddedDocs.length, "documents from website data.");
  app.listen(port, () => console.log(`Server running on port ${port}`));
});
