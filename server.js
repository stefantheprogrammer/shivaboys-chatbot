import express from "express";
import cors from "cors";
import { ChatGroq } from "@langchain/groq";
import { HuggingFaceInferenceEmbeddings } from "@langchain/community/embeddings/hf";
import { JSONLoader } from "langchain/document_loaders/fs/json";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { RetrievalQAChain } from "langchain/chains";
import * as fs from "fs";

const app = express();
const port = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

// === Load JSON website data as documents ===
const loader = new JSONLoader("data/website_data.json", {
  textKey: "content",
});
const rawDocs = await loader.load();

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 500,
  chunkOverlap: 50,
});
const splitDocs = await splitter.splitDocuments(rawDocs);

// === HuggingFace Embeddings (RAG) ===
const embeddings = new HuggingFaceInferenceEmbeddings({
  model: "sentence-transformers/all-MiniLM-L6-v2",
});
const vectorStore = await MemoryVectorStore.fromDocuments(splitDocs, embeddings);

// === Groq LLM (AI) ===
const chatModel = new ChatGroq({
  model: "llama3-8b-8192",
  apiKey: process.env.GROQ_API_KEY,
});

// === RAG Chain Setup ===
const chain = RetrievalQAChain.fromLLM(chatModel, vectorStore.asRetriever());

// === POST endpoint ===
app.post("/api/ask", async (req, res) => {
  const query = req.body.query;
  if (!query) {
    return res.status(400).json({ error: "Missing query" });
  }

  try {
    // === Try answering via RAG first ===
    const ragResult = await chain.call({ query });
    const ragAnswer = ragResult.text;

    // === Basic fallback detection ===
    const irrelevantResponses = [
      "I'm sorry, but I don't know the answer to that question.",
      "Sorry, I didn't understand that.",
      "I don't know the answer",
    ];

    const isRagRelevant = ragAnswer && !irrelevantResponses.some(msg => ragAnswer.toLowerCase().includes(msg.toLowerCase()));

    if (isRagRelevant) {
      return res.json({ answer: ragAnswer });
    }

    // === Fallback to Groq AI ===
    const aiResponse = await chatModel.invoke([
      { role: "user", content: query },
    ]);

    return res.json({ answer: aiResponse.content });

  } catch (error) {
    console.error("Error handling /api/ask:", error);
    res.status(500).json({ error: "Server error" });
  }
});

// === Home route ===
app.get("/", (req, res) => {
  res.send("Shiva Boys Chatbot backend is running.");
});

// === Start server ===
app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
