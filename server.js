import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import { ChatGroq } from '@langchain/groq';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import { loadQAStuffChain } from 'langchain/chains';
import { HuggingFaceTransformersEmbeddings } from '@langchain/community/embeddings/hf_transformers';

dotenv.config();
const app = express();
const port = process.env.PORT || 10000;

app.use(cors());
app.use(express.json());

let vectorStore;

async function fetchWebsiteText() {
  const res = await fetch('https://shivaboys.edu.tt');
  const html = await res.text();

  const plainText = html.replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ');
  return plainText.slice(0, 5000); // Optional: limit length for testing
}

async function loadDocuments() {
  console.log('Loading documents...');
  const rawText = await fetchWebsiteText();

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 50,
  });

  const docs = await splitter.createDocuments([rawText]);

  console.log(`Loaded ${docs.length} documents from website data.`);
  return docs;
}

async function createVectorStore(docs) {
  console.log('Computing embeddings...');

  const embeddings = new HuggingFaceTransformersEmbeddings({
    modelName: 'Xenova/all-MiniLM-L6-v2',
  });

  const store = await MemoryVectorStore.fromDocuments(docs, embeddings);
  return store;
}

async function main() {
  const documents = await loadDocuments();
  vectorStore = await createVectorStore(documents);
  console.log('Document embeddings ready.');
}

app.post('/ask', async (req, res) => {
  const question = req.body.question;

  if (!vectorStore) {
    return res.status(500).send('Vector store not initialized');
  }

  const relevantDocs = await vectorStore.similaritySearch(question, 5);

  const model = new ChatGroq({
    apiKey: process.env.GROQ_API_KEY,
    model: 'mixtral-8x7b-32768',
  });

  const chain = loadQAStuffChain(model);
  const response = await chain.call({
    input_documents: relevantDocs,
    question,
  });

  res.json({ answer: response.text });
});

app.listen(port, () => {
  console.log(`Server running on port ${port}`);
  main();
});
