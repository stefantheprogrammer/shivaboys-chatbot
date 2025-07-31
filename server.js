import express from 'express';
import cors from 'cors';
import cheerio from 'cheerio';
import https from 'https';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import { HuggingFaceTransformersEmbeddings } from '@langchain/community/embeddings/hf';
import { ChatGroq } from '@langchain/groq';
import { RetrievalQAChain } from 'langchain/chains';

const app = express();
const PORT = process.env.PORT || 10000;
app.use(cors());
app.use(express.json());

console.log('Server running on port', PORT);

// Step 1: Load website data
async function fetchWebsiteText(url) {
  const agent = new https.Agent({ rejectUnauthorized: false }); // Ignore SSL errors
  const res = await fetch(url, { agent });
  const html = await res.text();
  const $ = cheerio.load(html);
  return $('body').text();
}

async function loadDocuments() {
  console.log('Loading documents...');
  const pages = [
    'https://shivaboys.edu.tt',
    'https://shivaboys.edu.tt/about.html',
    'https://shivaboys.edu.tt/index.html',
    'https://shivaboys.edu.tt/department.html',
    'https://shivaboys.edu.tt/sixthform.html',
    'https://shivaboys.edu.tt/students.html',
    'https://shivaboys.edu.tt/curriculum.html',
    'https://shivaboys.edu.tt/parents.html',
    'https://shivaboys.edu.tt/achievements.html',
    'https://shivaboys.edu.tt/contact.html'
  ];

  const texts = await Promise.all(pages.map(fetchWebsiteText));
  return pages.map((url, i) => ({
    metadata: { source: url },
    pageContent: texts[i]
  }));
}

// Step 2: Create retriever
let retriever;

async function main() {
  const docs = await loadDocuments();

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200
  });

  const splitDocs = await splitter.splitDocuments(docs);

  const embeddings = new HuggingFaceTransformersEmbeddings({
    modelName: 'sentence-transformers/all-MiniLM-L6-v2'
  });

  const vectorStore = await MemoryVectorStore.fromDocuments(splitDocs, embeddings);

  retriever = vectorStore.asRetriever();
  console.log('Document embeddings ready.');
}

main();

// Step 3: QA Chain with Groq
const model = new ChatGroq({
  apiKey: process.env.GROQ_API_KEY,
  model: 'llama3-8b-8192'
});

const chain = RetrievalQAChain.fromLLM(model, retriever);

// Step 4: API endpoint
app.post('/chat', async (req, res) => {
  const { question } = req.body;
  try {
    const response = await chain.call({ query: question });
    res.json({ response: response.text });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Something went wrong.' });
  }
});

app.listen(PORT);
