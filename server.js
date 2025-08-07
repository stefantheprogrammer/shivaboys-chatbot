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
  // Load website data update HERE
const websiteData = JSON.parse(fs.readFileSync("data/website_data.json", "utf-8"));

// Load CSEC Math syllabus
const mathSyllabus = JSON.parse(fs.readFileSync("data/csec_maths_syllabus.json", "utf-8"));

// Combine both
const combinedData = [...websiteData, ...mathSyllabus];

// Convert to LangChain documents
const rawDocs = combinedData.map((entry) => {
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
const normalized = query.trim().toLowerCase();

  const greetings = {
    "hi": "Hi there! 👋 I’m Sage, the AI assistant for Shiva Boys’ Hindu College. How can I help you today?",
    "hello": "Hello! 😊 This is Sage from Shiva Boys’ Hindu College. What would you like to know?",
    "good morning": "Good morning! ☀️ I’m Sage, happy to assist you with anything about Shiva Boys’ Hindu College.",
    "good afternoon": "Good afternoon! 👋 I’m Sage. Let me know how I can help regarding the school.",
    "good evening": "Good evening! 👋 I’m Sage. Let me know how I can help regarding the school.",
  };

  if (greetings[normalized]) {
    return res.json({ answer: greetings[normalized] });
  }

  if (normalized === "who are you" || normalized === "where are you from") {
    return res.json({ answer: "I’m Sage, the AI assistant for Shiva Boys’ Hindu College in Trinidad and Tobago." });
  }
  // 🔹 Step 1: Normalize and check for quick keyword triggers
  const normalized = query.trim().toLowerCase();

  const quickTriggers = {
    "motto": "The motto for Shiva Boys' Hindu College is: 'Excellence, Duty, Truth'",
    "school motto": "The motto for Shiva Boys' Hindu College is: 'Excellence, Duty, Truth'",
    "location": "Shiva Boys' Hindu College is located at 35-37 Clarke Road, Penal, Trinidad & Tobago.",
    "address": "Shiva Boys' Hindu College is located at 35-37 Clarke Road, Penal, Trinidad & Tobago.",
    "phone": "Shiva Boys' Hindu College's phone number is (868)372-8822.",
    "contact": "Shiva Boys' Hindu College's phone number is (868)372-8822.",
    "email": "Shiva Boys' Hindu College's email address is ShivaBoys.sec@fac.edu.tt"
    
  };

  if (quickTriggers[normalized]) {
    return res.json({ answer: quickTriggers[normalized] });
  }

  try {
    // 🔹 Step 2: RAG (Vector-based search on your content)
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
  content: `
You are Sage — the official AI assistant for **Shiva Boys' Hindu College**, located at **35-37 Clarke Road, Penal, Trinidad & Tobago**.

Your job is to assist students, parents, and teachers with:
- Information about the school
- CXC CAPE and CXC CSEC syllabuses (Math, English A)
- Academic support
- Rules, news, and departments

Always introduce yourself as: 
"Hi, I’m Sage — the AI assistant for Shiva Boys' Hindu College."

❌ Never say you're from Barbados, or mention any other school.
❌ Never refer to yourself as an AI trained on public data, or say “according to the context.”
❌ Never guess the school name or location.
❌ Never say “based on the context.”
✅ Always speak naturally, clearly, and warmly — as if you're part of the school.

If you're unsure about something, say “I’m not sure about that. Would you like to check the school’s website or ask someone directly?”

Use Markdown formatting when helpful (e.g., lists, line breaks, bold) to make answers easy to read.

Always stay conversational and clear. You are not a search engine or a robot — you are Sage.`.trim()
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
