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

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const port = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

async function initialize() {
  try {
    const websiteData = JSON.parse(fs.readFileSync("data/website_data.json", "utf-8"));
    const mathSyllabus = JSON.parse(fs.readFileSync("data/csec_maths_syllabus.json", "utf-8"));
    const combinedData = [...websiteData, ...mathSyllabus];

    const rawDocs = combinedData.map(entry => new Document({
      pageContent: entry.content,
      metadata: { title: entry.title || "Untitled" },
    }));

    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 500,
      chunkOverlap: 50,
    });
    const splitDocs = await splitter.splitDocuments(rawDocs);

    const embeddings = new HuggingFaceInferenceEmbeddings({
      model: "sentence-transformers/all-MiniLM-L6-v2",
    });

    const vectorStore = await MemoryVectorStore.fromDocuments(splitDocs, embeddings);

    const chatModel = new ChatGroq({
      model: "llama3-8b-8192",
      apiKey: process.env.GROQ_API_KEY,
    });

    const chain = RetrievalQAChain.fromLLM(chatModel, vectorStore.asRetriever());

    async function performWebSearch(query) {
      const response = await fetch("https://api.search.brave.com/res/v1/web/search", {
        method: "POST",
        headers: {
          "Accept": "application/json",
          "Accept-Encoding": "gzip",
          "X-Subscription-Token": process.env.BRAVE_API_KEY,
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ q: query })
      });

      if (!response.ok) {
        throw new Error(`Brave Search API error: ${response.status}`);
      }

      const result = await response.json();

      if (!result.web || result.web.length === 0) {
        return "I searched the web, but couldn't find any relevant results.";
      }

      return result.web.slice(0, 3).map((r, i) =>
        `${i + 1}. [${r.title}](${r.url})\n${r.description}`
      ).join("\n\n");
    }

    app.post("/api/ask", async (req, res) => {
      const query = req.body.query;
      if (!query) return res.status(400).json({ error: "Missing query" });

      const normalized = query.trim().toLowerCase();

      // Predefined greetings
      const greetings = {
        "hi": "Hi there! 👋 I’m Sage, the AI assistant for Shiva Boys’ Hindu College. How can I help you today?",
        "hello": "Hello! 😊 This is Sage from Shiva Boys’ Hindu College. What would you like to know?",
        "good morning": "Good morning! ☀️ I’m Sage, happy to assist you with anything about Shiva Boys’ Hindu College.",
        "good afternoon": "Good afternoon! 👋 I’m Sage. Let me know how I can help regarding the school.",
        "good evening": "Good evening! 👋 I’m Sage. Let me know how I can help regarding the school."
      };
      if (greetings[normalized]) return res.json({ answer: greetings[normalized] });

      if (normalized === "who are you" || normalized === "where are you from") {
        return res.json({ answer: "I’m Sage, the AI assistant for Shiva Boys’ Hindu College in Trinidad and Tobago." });
      }

      // Quick reply triggers
      const quickTriggers = {
        "motto": "The motto for Shiva Boys' Hindu College is: 'Excellence, Duty, Truth'",
        "school motto": "The motto for Shiva Boys' Hindu College is: 'Excellence, Duty, Truth'",
        "location": "Shiva Boys' Hindu College is located at 35-37 Clarke Road, Penal, Trinidad & Tobago.",
        "address": "Shiva Boys' Hindu College is located at 35-37 Clarke Road, Penal, Trinidad & Tobago.",
        "phone": "Shiva Boys' Hindu College's phone number is (868)372-8822.",
        "contact": "Shiva Boys' Hindu College's phone number is (868)372-8822.",
        "email": "Shiva Boys' Hindu College's email address is ShivaBoys.sec@fac.edu.tt"
      };
      if (quickTriggers[normalized]) return res.json({ answer: quickTriggers[normalized] });

      try {
        // Use retrieval-augmented generation first
        const ragResult = await chain.call({ query });
        const ragAnswer = ragResult.text;

        const irrelevantResponses = [
          "I'm sorry, but I don't know the answer to that question.",
          "Sorry, I didn't understand that.",
          "I don't know the answer"
        ];

        const isRagRelevant = ragAnswer && !irrelevantResponses.some(msg =>
          ragAnswer.toLowerCase().includes(msg.toLowerCase())
        );

        if (isRagRelevant) return res.json({ answer: ragAnswer });

        // Setup system message for Groq
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

Always stay conversational and clear. You are not a search engine or a robot — you are Sage.
`.trim()
        };

        // Invoke Groq chat model
        const aiResponse = await chatModel.invoke([
          systemMessage,
          { role: "user", content: query }
        ]);

        const groqAnswer = aiResponse.content || "";

        // Check for weak/confident phrases in Groq's response
        const weakIndicators = [
          "according to my knowledge",
          "as of",
          "i believe",
          "i think",
          "possibly",
          "i'm not sure",
          "i'm unsure",
          "i don't have that information",
          "i don't know",
          "as of my knowledge cutoff"
        ];

        const isGroqWeak = weakIndicators.some(indicator =>
          groqAnswer.toLowerCase().includes(indicator)
        );

        if (!isGroqWeak) {
          // Groq gave a confident answer
          return res.json({ answer: groqAnswer });
        }

        // 🔍 Perform Brave Search as fallback
        try {
          const braveResults = await performWebSearch(query);
          return res.json({
            answer: `I couldn't answer confidently, so I searched the web for you:\n\n${braveResults}`
          });
        } catch (braveError) {
          console.error("Brave Search failed:", braveError.message);
          return res.json({
            answer: "I'm not sure about that, and I couldn't fetch live search results at the moment. Please try again later."
          });
        }

      } catch (error) {
        console.error("Error in /api/ask:", error);
        return res.status(500).json({ error: "Server error during question handling." });
      }
    });

    app.get("/", (req, res) => {
      res.send("Shiva Boys Chatbot backend is running.");
    });

    app.listen(port, () => {
      console.log(`✅ Server is running on port ${port}`);
    });

  } catch (err) {
    console.error("❌ Fatal startup error:", err.message);
    process.exit(1);
  }
}

initialize();
