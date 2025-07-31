import express from "express";
import dotenv from "dotenv";
import fetch from "node-fetch";
import cheerio from "cheerio";

import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { HNSWLib } from "langchain/vectorstores/hnswlib";
import { HuggingFaceInferenceEmbeddings } from "langchain/embeddings/huggingface";
import { GroqChat } from "@langchain/groq";
import { ConversationalRetrievalQAChain } from "langchain/chains";

dotenv.config();

const app = express();
const PORT = process.env.PORT || 10000;

app.use(express.json());

// Fetch and extract text from website (example: shivaboys.edu.tt)
async function fetchWebsiteText(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch ${url}: ${res.statusText}`);
  const html = await res.text();
  const $ = cheerio.load(html);
  return $("body").text().replace(/\s+/g, " ").trim();
}

async function loadDocuments() {
  // Example URL(s)
  const urls = ["https://shivaboys.edu.tt"];

  const texts = [];
  for (const url of urls) {
    const text = await fetchWebsiteText(url);
    texts.push(text);
  }

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 100,
  });

  const docs = await splitter.createDocuments(texts);
  return docs;
}

async function main() {
  console.log("Loading documents...");
  const docs = await loadDocuments();

  console.log("Creating vector store...");
  const vectorStore = await HNSWLib.fromDocuments(
    docs,
    new HuggingFaceInferenceEmbeddings({
      model: "hkunlp/instructor-large",
      apiKey: process.env.HUGGINGFACE_API_KEY,
    })
  );

  console.log("Initializing Groq chat...");
  const chat = new GroqChat({
    apiKey: process.env.GROQ_API_KEY,
    model: "gemma-7b-it",
    temperature: 0,
  });

  const chain = ConversationalRetrievalQAChain.fromLLM(chat, vectorStore.asRetriever());

  let conversationHistory = [];

  app.post("/chat", async (req, res) => {
    try {
      const { question } = req.body;
      if (!question) return res.status(400).json({ error: "Question is required" });

      const response = await chain.call({
        question,
        chat_history: conversationHistory,
      });

      conversationHistory.push([question, response.text]);

      res.json({ answer: response.text });
    } catch (error) {
      console.error("Error in /chat:", error);
      res.status(500).json({ error: "Internal Server Error" });
    }
  });

  app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
  });
}

main().catch((e) => {
  console.error("Fatal error:", e);
  process.exit(1);
});
