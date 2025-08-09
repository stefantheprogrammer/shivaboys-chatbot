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

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const logFilePath = path.join(__dirname, "chat_logs.txt");
const usageFilePath = path.join(__dirname, "usage_tracker.json");

// Limits — adjust as per your API limits or config
const limits = {
  brave: { daily: 1000, monthly: 30000 },
  bing: { daily: 1000, monthly: 30000 },
};

// Initialize usage data file if missing
function initializeUsageData() {
  if (!fs.existsSync(usageFilePath)) {
    const initData = {
      brave: { dailyCount: 0, monthlyCount: 0, lastReset: new Date().toDateString() },
      bing: { dailyCount: 0, monthlyCount: 0, lastReset: new Date().toDateString() },
    };
    fs.writeFileSync(usageFilePath, JSON.stringify(initData, null, 2));
    return initData;
  } else {
    return JSON.parse(fs.readFileSync(usageFilePath, "utf-8"));
  }
}

// Load usage data from file
function loadUsageData() {
  try {
    const data = fs.readFileSync(usageFilePath, "utf-8");
    return JSON.parse(data);
  } catch (e) {
    console.error("Failed to load usage data, initializing new.", e);
    return initializeUsageData();
  }
}

// Save usage data to file
function saveUsageData(data) {
  try {
    fs.writeFileSync(usageFilePath, JSON.stringify(data, null, 2));
  } catch (e) {
    console.error("Failed to save usage data.", e);
  }
}

// Reset usage counts daily and monthly if needed
function resetUsageIfNeeded(data) {
  const today = new Date().toDateString();
  if (data.brave.lastReset !== today) {
    data.brave.dailyCount = 0;
    data.bing.dailyCount = 0;
    data.brave.lastReset = today;
    data.bing.lastReset = today;
  }
  // You can also add monthly reset logic here if needed
  return data;
}

// Check quota function
function canUse(service, data) {
  return (
    data[service].dailyCount < limits[service].daily &&
    data[service].monthlyCount < limits[service].monthly
  );
}

// Increment usage count
function incrementUsage(service, data) {
  data[service].dailyCount++;
  data[service].monthlyCount++;
  saveUsageData(data);
}

const synonymMap = {
  sba: "school based assessment",
  "school based assessment": "school based assessment",
  csec: "caribbean secondary education certificate",
  cape: "caribbean advanced proficiency examination",
  "exam fees": "exam fees",
  fees: "fees",
  "tuition fees": "tuition fees",
  topic: "objective",
  objective: "topic",
};

function normalizeQuery(query) {
  let normalized = query.toLowerCase();
  for (const [alias, canonical] of Object.entries(synonymMap)) {
    const regex = new RegExp(`\\b${alias}\\b`, "gi");
    normalized = normalized.replace(regex, canonical);
  }
  return normalized;
}

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

const app = express();
const port = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

const conversationHistory = {};
function addToHistory(sessionId, role, content) {
  if (!conversationHistory[sessionId]) conversationHistory[sessionId] = [];
  conversationHistory[sessionId].push({ role, content });
  if (conversationHistory[sessionId].length > 10) {
    conversationHistory[sessionId] = conversationHistory[sessionId].slice(-10);
  }
}

const clarificationMap = {
  fees: "Are you asking about exam fees, registration fees, or tuition fees?",
  "subject combinations": "Do you mean combinations for CSEC, CAPE, or general school subjects?",
  timetable: "Are you interested in the daily timetable, exam timetable, or school events schedule?",
};

const clarificationAsked = {};

function isClarificationAnswered(originalClarificationKey, userQuery) {
  const clarificationsAnsweredExamples = {
    fees: ["exam fees", "tuition fees", "registration fees", "exam", "tuition", "registration"],
    "subject combinations": ["csec", "cape", "general subjects", "csec subjects", "cape subjects"],
    timetable: ["daily timetable", "exam timetable", "school events", "exam schedule"],
  };

  const normalizedQuery = userQuery.toLowerCase();
  if (!clarificationsAnsweredExamples[originalClarificationKey]) return false;

  return clarificationsAnsweredExamples[originalClarificationKey].some((example) =>
    normalizedQuery.includes(example)
  );
}

async function initialize() {
  try {
    // Load usage data at start
    let usageData = loadUsageData();
    usageData = resetUsageIfNeeded(usageData);
    saveUsageData(usageData);

    const dataFolder = path.join(__dirname, "data");
    if (!fs.existsSync(dataFolder)) {
      throw new Error(`Data folder not found at ${dataFolder}`);
    }

    const dataFiles = fs.readdirSync(dataFolder).filter((f) => f.endsWith(".json"));
    let combinedData = [];

    for (const file of dataFiles) {
      const filePath = path.join(dataFolder, file);
      try {
        const fileContent = JSON.parse(fs.readFileSync(filePath, "utf-8"));
        if (Array.isArray(fileContent)) {
          combinedData = combinedData.concat(fileContent);
          console.log(`Loaded ${file} (${fileContent.length} items)`);
        } else {
          if (fileContent && typeof fileContent === "object" && fileContent.content) {
            combinedData.push(fileContent);
            console.log(`Loaded ${file} (single object)`);
          } else {
            console.warn(`⚠️ Skipped ${file} — expected an array of documents or an object with {content}`);
          }
        }
      } catch (e) {
        console.error(`❌ Failed to parse ${file}:`, e.message);
      }
    }

    if (combinedData.length === 0) {
      console.warn("⚠️ No data loaded from /data — knowledge base will be empty.");
    } else {
      console.log(`✅ Combined data size: ${combinedData.length} entries`);
    }

    const rawDocs = combinedData.map((entry) =>
      new Document({ pageContent: entry.content, metadata: { title: entry.title || "Untitled" } })
    );

    const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 500, chunkOverlap: 50 });
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

    async function performBraveSearch(query) {
      if (!canUse("brave", usageData)) throw new Error("Brave Search quota exceeded");

      const response = await fetch(
        `https://api.search.brave.com/res/v1/web/search?q=${encodeURIComponent(query)}`,
        {
          method: "GET",
          headers: {
            Accept: "application/json",
            "X-Subscription-Token": process.env.BRAVE_API_KEY,
          },
        }
      );

      if (!response.ok) {
        if (response.status === 429) {
          throw new Error("Brave Search rate limit reached");
        }
        throw new Error(`Brave Search API error: ${response.status}`);
      }

      const result = await response.json();

      incrementUsage("brave", usageData);

      if (!result.web || !result.web.results || result.web.results.length === 0) {
        return "I searched the web, but couldn't find any relevant results.";
      }

      return result.web.results
        .slice(0, 3)
        .map((r, i) => `${i + 1}. [${r.title}](${r.url})\n${r.description}`)
        .join("\n\n");
    }

    async function performBingSearch(query) {
      if (!canUse("bing", usageData)) throw new Error("Bing Search quota exceeded");

      const url = `https://api.bing.microsoft.com/v7.0/search?q=${encodeURIComponent(query)}&count=3&textDecorations=true&textFormat=HTML`;

      const response = await fetch(url, {
        method: "GET",
        headers: {
          "Ocp-Apim-Subscription-Key": process.env.BING_API_KEY,
          Accept: "application/json",
        },
      });

      if (!response.ok) {
        if (response.status === 429) {
          throw new Error("Bing Search rate limit reached");
        }
        throw new Error(`Bing Search API error: ${response.status}`);
      }

      const result = await response.json();

      incrementUsage("bing", usageData);

      if (!result.webPages || !result.webPages.value || result.webPages.value.length === 0) {
        return "I searched the web, but couldn't find any relevant results.";
      }

      return result.webPages.value
        .slice(0, 3)
        .map(
          (r, i) =>
            `${i + 1}. [${r.name}](${r.url})\n${r.snippet.replace(/<[^>]*>/g, "")}`
        )
        .join("\n\n");
    }

    app.post("/api/ask", async (req, res) => {
      const queryRaw = req.body.query;
      const history = req.body.history || [];
      let sessionId = req.body.sessionId;

      if (!queryRaw) return res.status(400).json({ error: "Missing query" });

      if (!sessionId) sessionId = uuidv4();

      const normalized = normalizeQuery(queryRaw.trim());

      addToHistory(sessionId, "user", queryRaw);

      // Add your existing greeting, quick triggers, clarifications, personal facts logic here (omitted for brevity)

      try {
        const ragResult = await chain.call({ query: queryRaw });
        const ragAnswer = ragResult.text;

        const irrelevantResponses = [
          "i'm sorry",
          "i don't know",
          "sorry, i didn't understand",
        ];

        const isRagRelevant =
          ragAnswer &&
          !irrelevantResponses.some((msg) =>
            ragAnswer.toLowerCase().includes(msg)
          );

        if (isRagRelevant) {
          addToHistory(sessionId, "bot", ragAnswer);
          logChat(sessionId, queryRaw, ragAnswer);
          return res.json({ answer: ragAnswer, sessionId });
        }

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
❌ Never refer to yourself as an AI trained on public data.
❌ Never use phrases like “provided text,” “based on the context,” “according to the provided information,” or any similar wording.
❌ Never guess the school name or location.
✅ Always speak naturally, clearly, and warmly — as if you're part of the school.

If you're unsure about something or do not have specific information regarding Shiva Boys' Hindu College, reply with:

"I'm sorry, I don't have that information at the moment. For more details, please contact Shiva Boys' Hindu College directly at (868) 372-8822 or email ShivaBoys.sec@fac.edu.tt."

Would you like to check the school’s website or ask someone directly?
          `.trim(),
        };

        const chatModel = new ChatGroq({
          model: "llama3-8b-8192",
          apiKey: process.env.GROQ_API_KEY,
        });

        const groqResponse = await chatModel.invoke([...history, { role: "user", content: queryRaw }]);
        const groqAnswer = groqResponse.content || "";

        const weakIndicators = [
          "according to my knowledge",
          "as of",
          "i believe",
          "possibly",
          "i'm not sure",
          "i don't know",
          "i don't have that info",
          "i don't have any information",
          "i don't have that information",
        ];

        const isGroqWeak = weakIndicators.some((ind) =>
          groqAnswer.toLowerCase().includes(ind)
        );

        if (!isGroqWeak) {
          addToHistory(sessionId, "bot", groqAnswer);
          logChat(sessionId, queryRaw, groqAnswer);
          return res.json({ answer: groqAnswer, sessionId });
        }

        // Brave fallback
        try {
          if (!canUse("brave", usageData)) throw new Error("Brave Search quota exceeded");
          const braveResults = await performBraveSearch(queryRaw);
          const fallbackAnswer = `I couldn't answer confidently, so I searched the web for you:\n\n${braveResults}`;
          addToHistory(sessionId, "bot", fallbackAnswer);
          logChat(sessionId, queryRaw, fallbackAnswer);
          return res.json({ answer: fallbackAnswer, sessionId });
        } catch (braveError) {
          console.error("Brave Search failed or quota exceeded:", braveError.message);

          // Bing fallback
          try {
            if (!canUse("bing", usageData)) throw new Error("Bing Search quota exceeded");
            const bingResults = await performBingSearch(queryRaw);
            const fallbackAnswer = `I couldn't find a confident answer, so I searched the web for you:\n\n${bingResults}`;
            addToHistory(sessionId, "bot", fallbackAnswer);
            logChat(sessionId, queryRaw, fallbackAnswer);
            return res.json({ answer: fallbackAnswer, sessionId });
          } catch (bingError) {
            console.error("Bing Search failed or quota exceeded:", bingError.message);

            const finalFallback =
              "I'm sorry, I don't have that information at the moment. For more details, please contact Shiva Boys' Hindu College directly at (868) 372-8822 or email ShivaBoys.sec@fac.edu.tt.";
            addToHistory(sessionId, "bot", finalFallback);
            logChat(sessionId, queryRaw, finalFallback, bingError);
            return res.json({ answer: finalFallback, sessionId });
          }
        }
      } catch (error) {
        console.error("Error in /api/ask:", error);
        logChat(sessionId, queryRaw, null, error);
        res.status(500).json({ error: "Server error during question handling." });
      }
    });

    app.listen(port, () => {
      console.log(`Server running on port ${port}`);
    });
  } catch (err) {
    console.error("Error during setup:", err);
    process.exit(1);
  }
}

initialize();
