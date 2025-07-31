import express from 'express';
import cors from 'cors';
import fs from 'fs/promises';
import path from 'path';
import dotenv from 'dotenv';
import { fileURLToPath } from 'url';
import { pipeline } from '@xenova/transformers';
import { ChatGroq } from '@langchain/groq';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { RunnableSequence } from '@langchain/core/runnables';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { createRetrieverFromTexts } from 'langchain/retrievers/web';

dotenv.config();

const app = express();
const port = process.env.PORT || 10000;

app.use(cors());
app.use(express.json());

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// === EMBEDDINGS SETUP ===
let embedder = null;
const embeddingsCache = new Map();

async function embedText(text) {
  if (!embedder) {
    embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
  }

  if (embeddingsCache.has(text)) {
    return embeddingsCache.get(text);
  }

  const result = await embedder(text, { pooling: 'mean', normalize: true });
  const embedding = Array.from(result.data);
  embeddingsCache.set(text, embedding);
  return embedding;
}

// === LOAD DOCUMENTS ===
async function loadDocuments() {
  const dataDir = path.join(__dirname, 'data');
  const files = await fs.readdir(dataDir);
  const allDocs = [];

  for (const file of files) {
    const filePath = path.join(dataDir, file);
    const content = await fs.readFile(filePath, 'utf-8');
    try {
      const docs = JSON.parse(content);
      for (const d of docs) {
        if (d.title && d.content) {
          allDocs.push({ title: d.title, content: d.content });
        } else {
          console.warn(`⚠️ Skipped document in ${file} with missing title or content`);
        }
      }
    } catch (e) {
      console.warn(`⚠️ Failed to parse ${file}: ${e.message}`);
    }
  }

  return allDocs;
}

// === STARTUP LOGIC ===
let retriever;

async function initialize() {
  console.log('Loading documents...');
  const documents = await loadDocuments();
  console.log(`Loaded ${documents.length} documents from website data.`);

  const combinedText = documents.map(d => `Title: ${d.title}\nContent: ${d.content}`).join('\n\n');

  console.log('Computing embeddings for documents...');
  retriever = await createRetrieverFromTexts([combinedText], embedText, {
    k: 5,
  });

  console.log('Document embeddings ready.');
}

await initialize();

// === CHAT MODEL ===
const model = new ChatGroq({
  apiKey: process.env.GROQ_API_KEY,
  model: 'llama3-70b-8192',
});

const prompt = ChatPromptTemplate.fromMessages([
  ['system', 'You are a helpful assistant for a secondary school website. Answer questions using the context provided. If unsure, say "I\'m not sure about that."'],
  ['context', '{context}'],
  ['human', '{question}'],
]);

const chain = RunnableSequence.from([
  { question: (input) => input.question, context: async (input) => {
      const docs = await retriever.getRelevantDocuments(input.question);
      return docs.map(doc => doc.pageContent).join('\n\n');
    }
  },
  prompt,
  model,
  new StringOutputParser(),
]);

// === ROUTES ===
app.get('/', (req, res) => {
  res.send('Shiva Boys AI Chatbot is running.');
});

app.post('/chat', async (req, res) => {
  try {
    const { question } = req.body;
    const response = await chain.invoke({ question });
    res.json({ response });
  } catch (err) {
    console.error('Chat error:', err.message);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// === START SERVER ===
app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});
