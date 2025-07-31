import express from "express";
import dotenv from "dotenv";
import fetch from "node-fetch";
import cheerio from "cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { HNSWLib } from "langchain/vectorstores/hnswlib";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { ChatOpenAI } from "langchain/chat_models/openai";
import { ConversationalRetrievalQAChain } from "langchain/chains";

dotenv.config();

const app = express();
const PORT = process.env.PORT || 10000;

async function fetchWebsiteText(url) {
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Failed to fetch ${url}: ${res.statusText}`);
    const html = await res.text();
    const $ = cheerio.load(html);

    // Extract all text from the body, excluding scripts/styles
    const bodyText = $("body")
      .find("*")
      .not("script, style, noscript")
      .map((_, el) => $(el).text())
      .get()
      .join(" ");

    return bodyText.replace(/\s+/g, " ").trim();
  } catch (error) {
    console.error("Error fetching website:", error);
    return "";
  }
}

async function loadDocuments() {
  // Replace this URL with your actual school's website URL or a safe URL you want to scrape
  const url = "https://shivaboys.edu.tt";
  console.log("Fetching website content from:", url);

  const rawText = await fetchWebsiteText(url);
  if (!rawText) {
    throw new Error("Failed to fetch or parse website text.");
  }

  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });

  const docs = await textSplitter.splitText(rawText);
  return docs.map((text, i) => ({
    pageContent: text,
    metadata: { source: `${url}#chunk${i + 1}` },
  }));
}

async function main() {
  console.log("Loading documents...");
  const docs = await loadDocuments();

  console.log("Creating vector store...");
  const vectorStore = await HNSWLib.fromDocuments(docs, new OpenAIEmbeddings());

  console.log("Initializing chat model...");
  const model = new ChatOpenAI({
    temperature: 0,
    modelName: "gpt-4o-mini",
  });

  const chain = ConversationalRetrievalQAChain.fromLLM(model, vectorStore.asRetriever());

  app.use(express.json());

  let conversationHistory = [];

  app.post("/chat", async (req, res) => {
    try {
      const { question } = req.body;
      if (!question) {
        return res.status(400).json({ error: "Question is required" });
      }

      const response = await chain.call({
        question,
        chat_history: conversationHistory,
      });

      conversationHistory.push([question, response.text]);
      res.json({ answer: response.text });
    } catch (error) {
      console.error("Error in /chat:", error);
      res.status(500).json({ error: "Internal server error" });
    }
  });

  app.get("/", (req, res) => {
    res.send("Shiva Boys Chatbot Server is running.");
  });

  app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
  });
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});
