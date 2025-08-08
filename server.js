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
import fetch from "node-fetch";
import { v4 as uuidv4 } from "uuid";

// Get __dirname for ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const logFilePath = path.join(__dirname, "chat_logs.txt");

function logChat(sessionId, userQuery, assistantReply, error = null) {
  const timestamp = new Date().toISOString();
  const logEntry = {
    timestamp,
    sessionId: sessionId || "unknown",
    userQuery,
    assistantReply,
    error: error ? error.toString() : null,
  };
  try {
    fs.appendFileSync(logFilePath, JSON.stringify(logEntry) + "\n");
  } catch (err) {
    console.error("Failed to write chat log:", err);
  }
  console.log("Chat log:", JSON.stringify(logEntry));
}

// Improved vague input detection function
function isVagueInput(query) {
  const q = query.trim().toLowerCase();

  if (q.length <= 3) return true;

  const vagueWords = new Set([
    "fees", "fee", "subjects", "subject", "timetable", "schedule",
    "exam", "exams", "holidays", "vacation", "results", "marks",
    "grades", "library", "uniform", "transport", "canteen",
    "exam fees", "registration fees", "tuition fees"
  ]);

  // If query is a single vague word
  if (!q.includes(" ") && vagueWords.has(q)) return true;

  for (const vw of vagueWords) {
    if (q.includes(vw)) {
      const questionWords = ["what", "when", "where", "how", "why", "who", "which"];
      if (questionWords.some(w => q.includes(w))) {
        return false; // Not vague if question word present
      }
      return true; // Otherwise vague
    }
  }

  const questionWords = ["what", "when", "where", "how", "why", "who", "which"];
  if (!questionWords.some(w => q.includes(w))) return true;

  return false;
}

const app = express();
const port = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

// 🧠 Conversation memory by sessionId
const conversationHistory = {};
function addToHistory(sessionId, role, content) {
  if (!conversationHistory[sessionId]) {
    conversationHistory[sessionId] = [];
  }
  conversationHistory[sessionId].push({ role, content });
  if (conversationHistory[sessionId].length > 10) {
    conversationHistory[sessionId] = conversationHistory[sessionId].slice(-10);
  }
}

try {
  // Load data
  const websiteData = JSON.parse(fs.readFileSync("data/website_data.json", "utf-8"));
  const mathSyllabus = JSON.parse(fs.readFileSync("data/csec_maths_syllabus.json", "utf-8"));
  const combinedData = [...websiteData, ...mathSyllabus];
  const rawDocs = combinedData.map((entry) => new Document({
    pageContent: entry.content,
    metadata: { title: entry.title || "Untitled" },
  }));

  const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 500, chunkOverlap: 50 });
  const splitDocs = await splitter.splitDocuments(rawDocs);

  const embeddings = new HuggingFaceInferenceEmbeddings({ model: "sentence-transformers/all-MiniLM-L6-v2" });
  const vectorStore = await MemoryVectorStore.fromDocuments(splitDocs, embeddings);

  const chatModel = new ChatGroq({
    model: "llama3-8b-8192",
    apiKey: process.env.GROQ_API_KEY,
  });

  const chain = RetrievalQAChain.fromLLM(chatModel, vectorStore.asRetriever());

  // Brave Search for fallback
  async function performWebSearch(query) {
    const response = await fetch(`https://api.search.brave.com/res/v1/web/search?q=${encodeURIComponent(query)}`, {
      method: "GET",
      headers: {
        "Accept": "application/json",
        "X-Subscription-Token": process.env.BRAVE_API_KEY
      }
    });
    if (!response.ok) {
      throw new Error(`Brave Search API error: ${response.status}`);
    }
    const result = await response.json();
    if (!result.web || !result.web.results || result.web.results.length === 0) {
      return "I searched the web, but couldn't find any relevant results.";
    }
    return result.web.results.slice(0, 3).map((r, i) =>
      `${i + 1}. [${r.title}](${r.url})\n${r.description}`
    ).join("\n\n");
  }

  // Expanded quickTriggers with exam fees etc.
  const quickTriggers = {
    "hi": "Hi there! 👋 I’m Sage, the AI assistant for Shiva Boys’ Hindu College. How can I help you today?",
    "hello": "Hello! 😊 This is Sage from Shiva Boys’ Hindu College. What would you like to know?",
    "good morning": "Good morning! ☀️ I’m Sage, happy to assist you with anything about Shiva Boys’ Hindu College.",
    "good afternoon": "Good afternoon! 👋 I’m Sage. Let me know how I can help regarding the school.",
    "good evening": "Good evening! 👋 I’m Sage. Let me know how I can help regarding the school.",

    "fees": "Are you asking about exam fees, registration fees, or tuition fees?",
    "fee": "Are you asking about exam fees, registration fees, or tuition fees?",
    "exam fees": "Exam fees vary depending on the exam board and subject. Would you like the exact amounts or payment deadlines?",
    "registration fees": "Registration fees for the exams depend on the exam board and year. I can help with specific details if you like.",
    "tuition fees": "Tuition fees depend on the grade and subjects. Please specify which grade or subject you want to know about.",

    "motto": "The motto for Shiva Boys' Hindu College is: 'Excellence, Duty, Truth'",
    "school motto": "The motto for Shiva Boys' Hindu College is: 'Excellence, Duty, Truth'",
    "location": "Shiva Boys' Hindu College is located at 35-37 Clarke Road, Penal, Trinidad & Tobago.",
    "address": "Shiva Boys' Hindu College is located at 35-37 Clarke Road, Penal, Trinidad & Tobago.",
    "phone": "Shiva Boys' Hindu College's phone number is (868)372-8822.",
    "contact": "Shiva Boys' Hindu College's phone number is (868)372-8822.",
    "email": "Shiva Boys' Hindu College's email address is ShivaBoys.sec@fac.edu.tt"
  };

  // Controlled Personal Facts
  const personalFacts = {
    principal: {
      keywords: ["principal", "headmaster", "school principal"],
      name: "Mr. Devinesh Neeranjan",
      description: "the Principal of Shiva Boys' Hindu College",
      comment: "Yes, his name is quite unique and lovely."
    },
  };

  function checkPersonalFacts(q) {
    const qNorm = q.toLowerCase();
    for (const key in personalFacts) {
      const fact = personalFacts[key];
      if (fact.keywords.some(kw => qNorm.includes(kw))) {
        return `The ${fact.description} is ${fact.name}.\n\n${fact.comment}`;
      }
    }
    return null;
  }

  app.post("/api/ask", async (req, res) => {
    let { query, history, sessionId } = req.body;
    if (!query) {
      return res.status(400).json({ error: "Missing query" });
    }
    query = query.trim();
    sessionId = sessionId || uuidv4(); // auto-generate sessionId if missing
    history = history || [];

    const normalized = query.toLowerCase();

    // Quick triggers (greetings, quick answers)
    if (quickTriggers[normalized]) {
      logChat(sessionId, query, quickTriggers[normalized]);
      return res.json({ answer: quickTriggers[normalized], sessionId });
    }

    // Check personal facts
    const factResponse = checkPersonalFacts(query);
    if (factResponse) {
      try {
        const systemMsg = {
          role: "system",
          content: `You are Sage, the friendly AI assistant for Shiva Boys' Hindu College. Respond warmly and naturally.`
        };
        const userMsg = {
          role: "user",
          content: factResponse
        };
        const creativeReply = await chatModel.invoke([systemMsg, userMsg]);
        logChat(sessionId, query, creativeReply.content);
        return res.json({ answer: creativeReply.content, sessionId });
      } catch (err) {
        console.error("Error generating creative reply:", err);
        logChat(sessionId, query, factResponse, err);
        return res.json({ answer: factResponse, sessionId });
      }
    }

    // Vague input detection
    if (isVagueInput(query)) {
      const clarification = "I want to make sure I understand — could you please provide more details or specify your question?";
      logChat(sessionId, query, clarification);
      return res.json({ answer: clarification, sessionId });
    }

    // Try RAG chain first
    try {
      const ragResult = await chain.call({ query });
      const ragAnswer = ragResult.text;

      const irrelevantResponses = [
        "i'm sorry", "i don't know", "sorry, i didn't understand"
      ];

      const isRagRelevant = ragAnswer && !irrelevantResponses.some(msg =>
        ragAnswer.toLowerCase().includes(msg)
      );

      if (isRagRelevant) {
        logChat(sessionId, query, ragAnswer);
        return res.json({ answer: ragAnswer, sessionId });
      }

      // Use Groq LLM fallback
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
✅ Always speak naturally, clearly, and warmly — as if you're part of the school.

If you're unsure about something, say:
“I’m not sure about that. Would you like to check the school’s website or ask someone directly?”
        `.trim()
      };

      const groqResponse = await chatModel.invoke([
        systemMessage,
        ...history,
        { role: "user", content: query }
      ]);

      const groqAnswer = groqResponse.content || "";

      const weakIndicators = [
        "according to my knowledge", "as of", "i believe", "possibly", "i'm not sure", "i don't know"
      ];

      const isGroqWeak = weakIndicators.some(ind =>
        groqAnswer.toLowerCase().includes(ind)
      );

      if (!isGroqWeak) {
        logChat(sessionId, query, groqAnswer);
        return res.json({ answer: groqAnswer, sessionId });
      }

      // Brave Search fallback
      try {
        const braveResults = await performWebSearch(query);
        const fallbackAnswer = `I couldn't answer confidently, so I searched the web for you:\n\n${braveResults}`;
        logChat(sessionId, query, fallbackAnswer);
        return res.json({ answer: fallbackAnswer, sessionId });
      } catch (braveError) {
        console.error("Brave Search failed:", braveError.message);
        const fallbackFailAnswer = "I'm not sure about that, and I couldn't fetch live search results at the moment. Please try again later.";
        logChat(sessionId, query, fallbackFailAnswer, braveError);
        return res.json({ answer: fallbackFailAnswer, sessionId });
      }
    } catch (error) {
      console.error("Error in /api/ask:", error);
      logChat(sessionId, query, null, error);
      res.status(500).json({ error: "Server error during question handling." });
    }
  });

} catch (err) {
  console.error("Error during setup:", err);
  process.exit(1);
}

app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});
