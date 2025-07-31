import express from 'express';
import cors from 'cors';
import morgan from 'morgan';
import dotenv from 'dotenv';
import fs from 'fs/promises';
import path from 'path';
import { HuggingFaceInference } from '@huggingface/inference';
import { HuggingFaceTransformersEmbeddings } from 'langchain/embeddings/hf';
import { RetrievalQAChain } from 'langchain/chains';
import { VectorStoreRetriever } from 'langchain/vectorstores/base';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import { TextLoader } from 'langchain/document_loaders/fs/text';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';

dotenv.config();
const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(morgan('dev'));
app.use(express.json());

const hf = new HuggingFaceInference(process.env.HUGGINGFACE_API_KEY);

const embedder = new HuggingFaceTransformersEmbeddings({
  model: "sentence-transformers/all-MiniLM-L6-v2",
});

let qaChain;

async function init() {
  const dataDir = './data';
  const files = await fs.readdir(dataDir);

  const docs = [];
  for (const file of files) {
    const filePath = path.join(dataDir, file);
    const loader = new TextLoader(filePath);
    const loadedDocs = await loader.load();
    docs.push(...loadedDocs);
  }

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 150,
  });

  const splitDocs = await splitter.splitDocuments(docs);
  const vectorStore = await MemoryVectorStore.fromDocuments(splitDocs, embedder);
  const retriever = new VectorStoreRetriever({ vectorStore });

  qaChain = RetrievalQAChain.fromLLM(
    {
      llm: {
        _call: async ({ prompt }) => {
          const res = await hf.textGeneration({
            model: "mistralai/Mixtral-8x7B-Instruct-v0.1",
            inputs: prompt,
            parameters: { max_new_tokens: 300, temperature: 0.7 },
          });
          return res.generated_text || res[0]?.generated_text || "Sorry, I couldn't understand.";
        }
      }
    },
    retriever
  );
}

app.post('/ask', async (req, res) => {
  try {
    const question = req.body.question;
    const result = await qaChain.call({ query: question });
    res.json({ answer: result.text });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Something went wrong' });
  }
});

app.get('/', (_, res) => {
  res.send('Shiva Boys Chatbot API is running.');
});

init().then(() => {
  app.listen(PORT, () => console.log(`✅ Server running on port ${PORT}`));
});
