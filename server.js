import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import { ChatGroq } from "@langchain/groq";
import { HuggingFaceInferenceEmbeddings } from "@langchain/community/embeddings/hf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { RetrievalQAChain } from "langchain/chains";
import { Document } from "@langchain/core/documents";
import fetch from "node-fetch";

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());

const PORT = process.env.PORT || 10000;
let chain;

async function fetchWebsiteText(url) {
  const response = await fetch(url);
  const html = await response.text();
  return html.replace(/<[^>]*>/g, " ").replace(/\s+/g, " ");
}

async function loadDocuments() {
  const urls = [
    "https://shivaboys.edu.tt/index.html",
    "https://shivaboys.edu.tt/about.html",
    "https://shivaboys.edu.tt/administration.html",
    "https://shivaboys.edu.tt/departments.html",
    "https://shivaboys.edu.tt/registration.html"
  ];
  const documents = [];
  for (const url of urls) {
    const text = await fetchWebsiteText(url);
    documents.push(new Document({ pageContent: text, metadata: { source: url } }));
  }
  return documents;
}

async function main() {
  const docs = await loadDocuments();
  const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000, chunkOverlap: 200 });
  const splitDocs = await splitter.splitDocuments(docs);

  const vectorstore = await MemoryVectorStore.fromDocuments(
    splitDocs,
    new HuggingFaceInferenceEmbeddings({
      apiKey: process.env.HF_API_KEY,
      model: "sentence-transformers/all-MiniLM-L6-v2"
    })
  );

  const model = new ChatGroq({
    apiKey: process.env.GROQ_API_KEY,
    model: "llama3-8b-8192"
  });

  chain = RetrievalQAChain.fromLLM(model, vectorstore.asRetriever());
}

app.post("/chat", async (req, res) => {
  try {
    const { question } = req.body;
    if (!question) return res.status(400).json({ error: "Missing question" });

    const response = await chain.call({ query: question });
    res.json({ response: response.text });
  } catch (err) {
    console.error("Error:", err.message);
    res.status(500).json({ error: "Something went wrong" });
  }
});

main()
  .then(() => app.listen(PORT, () => console.log(`✅ Server running on port ${PORT}`)))
  .catch(err => console.error("Startup error:", err));
