import express from "express";
import cors from "cors";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { ChatGroq } from "@langchain/groq";
import { HuggingFaceInferenceEmbeddings } from "@langchain/community/embeddings/hf";
import { RetrievalQAChain } from "langchain/chains";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { Document } from "@langchain/core/documents";

// Helpers to resolve directory paths
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(cors());
app.use(express.json());

// ✅ Serve widget.html and other static files from /public
app.use(express.static("public"));

// ✅ Load website content (RAG data)
const raw = fs.readFileSync(path.join(__dirname, "data", "website_data.json"), "utf-8");
const pages = JSON.parse(raw);

// ✅ Create documents from your pages
const documents = pages.map((page) => new Document({ pageContent: page.content, metadata: { source: page.title } }));

// ✅ Split documents
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
});
const splitDocs = await splitter.splitDocuments(documents);

// ✅ Embed using HuggingFace
const embeddings = new HuggingFaceInferenceEmbeddings({
  apiKey: process.env.HUGGINGFACE_API_KEY,
  model: "sentence-transformers/all-mpnet-base-v2",
});

// ✅ Store in-memory vector DB
const vectorStore = await MemoryVectorStore.fromDocuments(splitDocs, embeddings);

// ✅ Set up retriever
const retriever = vectorStore.asRetriever();

// ✅ Set up Groq chat model (for AI + RAG)
const chatModel = new ChatGroq({
  apiKey: process.env.GROQ_API_KEY,
  model: "llama3-8b-8192",
});

// ✅ Use a hybrid RAG QA Chain
const chain = RetrievalQAChain.fromLLM(chatModel, retriever, {
  returnSourceDocuments: true,
});

// ✅ Handle incoming questions
app.post("/api/ask", async (req, res) => {
  const question = req.body.question;
  if (!question || question.trim() === "") {
    return res.status(400).json({ error: "Question is required." });
  }

  try {
    const result = await chain.invoke({ query: question });

    if (!result || !result.text) {
      return res.json({ answer: "Sorry, I didn't understand that." });
    }

    res.json({ answer: result.text });
  } catch (err) {
    console.error("Error:", err);
    res.status(500).json({ error: "An error occurred while processing your request." });
  }
});

// ✅ Start the server
const port = process.env.PORT || 3000;
app.listen(port, () => {
  console.log(`✅ Server running on port ${port}`);
});
