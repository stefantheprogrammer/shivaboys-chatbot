import express from "express";
import * as cheerio from "cheerio";
import * as dotenv from "dotenv";
import { ChatGroq } from "@langchain/groq";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { Embeddings } from "langchain/embeddings/base";
import { RetrievalQAChain } from "langchain/chains";
import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";

dotenv.config();

const app = express();
const port = process.env.PORT || 10000;
let chain;

app.use(express.json());

// Use HuggingFace free embeddings (MiniLM)
class HuggingFaceEmbeddings extends Embeddings {
  async embedDocuments(documents) {
    const res = await fetch("https://api-inference.huggingface.co/embeddings/sentence-transformers/all-MiniLM-L6-v2", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${process.env.HUGGINGFACE_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ inputs: documents })
    });
    const json = await res.json();
    return json;
  }

  async embedQuery(query) {
    return (await this.embedDocuments([query]))[0];
  }
}

// Scrape website content
async function fetchWebsiteText(url) {
  try {
    const loader = new CheerioWebBaseLoader(url);
    const docs = await loader.load();
    return docs;
  } catch (error) {
    console.error("Failed to fetch:", url, error);
    return [];
  }
}

// Load and embed documents
async function loadDocuments() {
  const pages = [
    "https://shivaboys.edu.tt/",
    "https://shivaboys.edu.tt/about.html",
    "https://shivaboys.edu.tt/index.html",
    "https://shivaboys.edu.tt/department.html",
    "https://shivaboys.edu.tt/curriculum.html",
    "https://shivaboys.edu.tt/csec.html"
  ];

  let allDocs = [];
  for (const page of pages) {
    const docs = await fetchWebsiteText(page);
    allDocs.push(...docs);
  }

  const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000, chunkOverlap: 200 });
  const splitDocs = await splitter.splitDocuments(allDocs);

  const vectorStore = await MemoryVectorStore.fromDocuments(splitDocs, new HuggingFaceEmbeddings());
  return vectorStore;
}

// Initialize vector store and chain
async function main() {
  console.log("Loading documents...");
  const vectorStore = await loadDocuments();

  const model = new ChatGroq({
    apiKey: process.env.GROQ_API_KEY,
    model: "llama3-8b-8192"
  });

  chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever());
  console.log("Chatbot ready.");
}

app.post("/chat", async (req, res) => {
  const question = req.body.question;
  if (!question || !chain) {
    return res.status(400).json({ error: "No question or model not ready." });
  }

  try {
    const response = await chain.call({ query: question });
    res.json({ response: response.text });
  } catch (err) {
    console.error("Error:", err);
    res.status(500).json({ error: "Something went wrong." });
  }
});

app.listen(port, () => {
  console.log(`Server running on port ${port}`);
  main();
});