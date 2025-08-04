import express from "express";
import cors from "cors";
import path from "path";
import { fileURLToPath } from "url";
import { ChatGroq } from "@langchain/groq";
import { HuggingFaceInferenceEmbeddings } from "@langchain/community/embeddings/hf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { RetrievalQAChain } from "langchain/chains";
import { Document } from "@langchain/core/documents";
import * as fs from "fs";

// Get __dirname in ES Modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const port = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

// Serve static files (like widget.html) from /public folder
app.use(express.static(path.join(__dirname, "public")));

try {
  // Load and parse your JSON website data manually
  const websiteData = JSON.parse(fs.readFileSync("data/website_data.json", "utf-8"));
  const rawDocs = websiteData.map((entry) => {
    return new Document({
      pageContent: entry.content,
      metadata: { title: entry.title || "Untitled" },
    });
  });

  // Split documents into chunks
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 50,
  });
  const splitDocs = await splitter.splitDocuments(rawDocs);

  // Create embeddings
  const embeddings = new HuggingFaceInferenceEmbeddings({
    model: "sentence-transformers/all-MiniLM-L6-v2",
  });
  const vectorStore = await MemoryVectorStore.fromDocuments(splitDocs, embeddings);

  // Initialize Groq chat model
  const chatModel = new ChatGroq({
    model: "llama3-8b-8192",
    apiKey: process.env.GROQ_API_KEY,
  });

  // Setup RetrievalQA chain
  const chain = RetrievalQAChain.fromLLM(chatModel, vectorStore.asRetriever());

  // POST /api/ask endpoint
  app.post("/api/ask", async (req, res) => {
    const query = req.body.query;
    if (!query) {
      return res.status(400).json({ error: "Missing query" });
    }

    try {
      const ragResult = await chain.call({ query });
      const ragAnswer = ragResult.text;

      const irrelevantResponses = [
        "I'm sorry, but I don't know the answer to that question.",
        "Sorry, I didn't understand that.",
        "I don't know the answer",
      ];

      const isRagRelevant = ragAnswer && !irrelevantResponses.some(msg =>
        ragAnswer.toLowerCase().includes(msg.toLowerCase())
      );

      if (isRagRelevant) {
        return res.json({ answer: ragAnswer });
      }

      // Fallback to direct LLM chat
const systemMessage = {
  role: "system",
  content: "You are Sage, the AI assistant for Shiva Boys' Hindu College. Use the provided website information to answer questions clearly and naturally. If you don't know something based on the context, say so. When helpful, refer to the information source with phrases like "According to the school's website" or "Based on the information I have." Do not make up information."
};

const aiResponse = await chatModel.invoke([
  systemMessage,
  { role: "user", content: query }
]);
return res.json({ answer: aiResponse.content });

    } catch (error) {
      console.error("Error in /api/ask:", error);
      res.status(500).json({ error: "Server error during question handling." });
    }
  });

  // Health check root endpoint
  app.get("/", (req, res) => {
    res.send("Shiva Boys Chatbot backend is running.");
  });

  // Start the server
  app.listen(port, () => {
    console.log(`✅ Server is running on port ${port}`);
  });

} catch (err) {
  console.error("❌ Fatal startup error:", err.message);
  process.exit(1);
}
