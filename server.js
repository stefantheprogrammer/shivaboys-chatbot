import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import fs from "fs";
import { ChatGroq } from "@langchain/groq";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/hf_transformers";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { RunnableSequence } from "@langchain/core/runnables";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { pull } from "langchain/hub";

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());

const PORT = process.env.PORT || 10000;

// Load website data
const rawData = fs.readFileSync('./data/website_data.json', 'utf8');
const pages = JSON.parse(raw);

// Prepare documents
const documents = pages.map((page) => ({
  pageContent: `${page.title}\n\n${page.content}`,
  metadata: { source: page.title },
}));

// Split documents into chunks
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 100,
});
const splitDocs = await splitter.splitDocuments(documents);

// Create vector store
const embeddings = new HuggingFaceTransformersEmbeddings();
const store = await MemoryVectorStore.fromDocuments(splitDocs, embeddings);

// Load prompt from LangChain hub
const prompt = await pull("rlm/rag-prompt");

// Setup chat model and chain
const model = new ChatGroq({
  apiKey: process.env.GROQ_API_KEY,
  model: "llama3-8b-8192",
});

const retriever = store.asRetriever();
const chain = await RunnableSequence.from([
  {
    context: retriever.pipe(docs => docs.map(doc => doc.pageContent).join("\n\n")),
    question: input => input.query,
  },
  prompt,
  model,
  new StringOutputParser(),
]);

// Serve static widget files
app.use("/", express.static("public"));

// RAG API endpoint
app.post("/api/ask", async (req, res) => {
  const question = req.body.message;
  if (!question) return res.status(400).json({ error: "Missing message" });

  try {
    let answer = "";
    const response = await chain.invoke({ query: question });
    answer = response;

    // Fallback to LLM-only response if context-based answer is empty or vague
    if (
      !answer ||
      answer.toLowerCase().includes("i don't know") ||
      answer.toLowerCase().includes("does not provide") ||
      answer.toLowerCase().includes("not available")
    ) {
      const fallback = await model.invoke(`Answer the following as best you can:\n\n${question}`);
      answer = fallback.content;
    }

    res.json({ answer });
  } catch (err) {
    console.error("Error:", err);
    res.status(500).json({ error: "Something went wrong." });
  }
});

// Start server
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
