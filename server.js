import express from "express";
import cors from "cors";
import morgan from "morgan";
import dotenv from "dotenv";
import { HuggingFaceInferenceEmbeddings } from "langchain/embeddings/hf";
import { ChatHuggingFaceInference } from "langchain/chat_models/hf";
import { RetrievalQAChain } from "langchain/chains";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import fs from "fs/promises";
import path from "path";
import { fileURLToPath } from "url";

dotenv.config();
const app = express();
const port = process.env.PORT || 3000;

app.use(cors());
app.use(morgan("dev"));
app.use(express.json());

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let qaChain;

async function init() {
  const embeddings = new HuggingFaceInferenceEmbeddings({
    model: "sentence-transformers/all-MiniLM-L6-v2",
    apiKey: process.env.HUGGINGFACE_API_KEY,
  });

  const loaderPath = path.join(__dirname, "data");
  const files = await fs.readdir(loaderPath);

  let docs = [];
  for (const file of files) {
    const content = await fs.readFile(path.join(loaderPath, file), "utf-8");
    docs.push({ pageContent: content });
  }

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 50,
  });

  const splitDocs = await splitter.splitDocuments(docs);
  const vectorStore = await MemoryVectorStore.fromDocuments(splitDocs, embeddings);

  const retriever = vectorStore.asRetriever();
  const model = new ChatHuggingFaceInference({
    model: "mistralai/Mixtral-8x7B-Instruct-v0.1",
    apiKey: process.env.HUGGINGFACE_API_KEY,
  });

  qaChain = RetrievalQAChain.fromLLM(model, retriever);
}

app.post("/chat", async (req, res) => {
  try {
    const question = req.body.question;
    const response = await qaChain.call({ query: question });
    res.json({ response: response.text });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Something went wrong." });
  }
});

init().then(() => {
  app.listen(port, () => {
    console.log(`✅ Server running on port ${port}`);
  });
});
