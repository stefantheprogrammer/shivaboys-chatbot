import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import fs from "fs";

import { ChatGroq } from "@langchain/groq";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/huggingface_transformers";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { RetrievalQAChain } from "langchain/chains";

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());

const PORT = process.env.PORT || 3000;

// Load documents for retrieval
const raw = fs.readFileSync("data/website_data.json", "utf-8");
const pages = JSON.parse(raw);
const docs = pages.map((p) => ({
  pageContent: p.content,
  metadata: { source: p.title || "Untitled" },
}));

// Setup embeddings and vector store
const embeddings = new HuggingFaceTransformersEmbeddings({
  modelName: "Xenova/all-MiniLM-L6-v2",
});

const vectorStore = await MemoryVectorStore.fromTexts(
  docs.map((d) => d.pageContent),
  docs.map((d) => d.metadata),
  embeddings
);

// Setup retrieval QA chain
const retriever = vectorStore.asRetriever();

const retrievalChain = RetrievalQAChain.fromLLM(
  new ChatGroq({
    apiKey: process.env.GROQ_API_KEY,
    model: "llama3-8b-8192",
  }),
  retriever
);

// Setup general AI LLM for fallback
const groqLLM = new ChatGroq({
  apiKey: process.env.GROQ_API_KEY,
  model: "llama3-8b-8192",
});

app.post("/api/ask", async (req, res) => {
  try {
    const { question } = req.body;

    // Step 1: Try retrieval QA
    const retrievalResponse = await retrievalChain.call({ query: question });

    // If the answer is short, vague, or not useful, fallback to general AI
    if (
      !retrievalResponse.text ||
      retrievalResponse.text.toLowerCase().includes("don't know") ||
      retrievalResponse.text.length < 30
    ) {
      // Step 2: Fallback to Groq general LLM for freeform answer
      const aiResponse = await groqLLM.call(question);
      return res.json({ answer: aiResponse });
    }

    // Step 3: Otherwise send retrieval answer
    res.json({ answer: retrievalResponse.text });
  } catch (error) {
    console.error("Error in /api/ask:", error);
    res.status(500).json({ answer: "Sorry, something went wrong." });
  }
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
