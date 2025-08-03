import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import { ChatGroq } from "@langchain/groq";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/huggingface_transformers";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { RetrievalQAChain } from "langchain/chains";
import fs from "fs";

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());

// Add this line to serve static files from 'public' folder:
app.use(express.static("public"));

const PORT = process.env.PORT || 3000;

// Load documents
const data = JSON.parse(fs.readFileSync("data/website_data.json", "utf-8"));
const docs = data.map((item) => ({
  pageContent: item.content,
  metadata: { source: item.title || "Untitled" },
}));

// Set up embeddings
const embeddings = new HuggingFaceTransformersEmbeddings({
  modelName: "Xenova/all-MiniLM-L6-v2",
});

const vectorStore = await MemoryVectorStore.fromTexts(
  docs.map((d) => d.pageContent),
  docs.map((d) => d.metadata),
  embeddings
);

// Set up Groq chat model
const model = new ChatGroq({
  apiKey: process.env.GROQ_API_KEY,
  model: "llama3-8b-8192",
});

const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever());

app.post("/api/ask", async (req, res) => {
  try {
    const { question } = req.body;
    const response = await chain.call({ query: question });
    res.json({ answer: response.text });
  } catch (error) {
    console.error("Error:", error);
    res.status(500).json({ error: "Something went wrong" });
  }
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
