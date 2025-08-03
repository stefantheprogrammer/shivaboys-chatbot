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

// Load and prepare documents
const data = JSON.parse(fs.readFileSync("data/website_data.json", "utf-8"));
const docs = data.map((item) => ({
  pageContent: item.content,
  metadata: { source: item.title || "Untitled" },
}));

// Set up embeddings
const embeddings = new HuggingFaceTransformersEmbeddings({
  modelName: "Xenova/all-MiniLM-L6-v2",
});

// Create vector store from documents
const vectorStore = await MemoryVectorStore.fromTexts(
  docs.map((d) => d.pageContent),
  docs.map((d) => d.metadata),
  embeddings
);

// Initialize Groq chat model
const model = new ChatGroq({
  apiKey: process.env.GROQ_API_KEY,
  model: "llama3-8b-8192",
});

// Create retrieval QA chain
const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever());

app.post("/api/ask", async (req, res) => {
  try {
    const { question } = req.body;

    // Try retrieval QA first
    const response = await chain.call({ query: question });

    if (
      response.text && 
      (!response.sourceDocuments || response.sourceDocuments.length === 0)
    ) {
      // If no source documents found, fallback to plain LLM query
      const fallbackResponse = await model.invoke(question);
      return res.json({ answer: fallbackResponse.text || fallbackResponse });
    }

    // Respond with retrieval answer
    res.json({ answer: response.text });
  } catch (error) {
    console.error("Error in /api/ask:", error);
    res.status(500).json({ error: "Something went wrong" });
  }
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
