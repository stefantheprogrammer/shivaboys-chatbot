import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/hf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { ChatGroq } from "@langchain/groq";
import { RetrievalQAChain } from "langchain/chains";
import { BufferMemory } from "langchain/memory";

dotenv.config();

const app = express();
const port = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

let vectorStore = null;

// Load docs into memory
async function loadDocs() {
  const docs = [
    {
      title: "Welcome",
      content: "Shiva Boys’ Hindu College is a government-assisted secondary school located in Trinidad and Tobago..."
    },
    {
      title: "Curriculum",
      content: "The curriculum includes Mathematics, English, Science, Information Technology, Business Studies, and Modern Languages..."
    },
    // Add more documents as needed
  ];

  const formattedDocs = docs.map(doc => ({
    pageContent: doc.content,
    metadata: { title: doc.title }
  }));

  const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 500, chunkOverlap: 50 });
  const splitDocs = await splitter.splitDocuments(formattedDocs);

  const embeddings = new HuggingFaceTransformersEmbeddings({
    modelName: "sentence-transformers/all-MiniLM-L6-v2"
  });

  vectorStore = await MemoryVectorStore.fromDocuments(splitDocs, embeddings);
}

await loadDocs();

app.post("/chat", async (req, res) => {
  const { message } = req.body;

  try {
    const model = new ChatGroq({
      apiKey: process.env.GROQ_API_KEY,
      model: "llama3-8b-8192",
      temperature: 0.7,
    });

    const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever(), {
      memory: new BufferMemory(),
      returnSourceDocuments: true,
    });

    const response = await chain.call({ query: message });

    res.json({ response: response.text });
  } catch (error) {
    console.error("Chat error:", error);
    res.status(500).json({ error: "Failed to generate response" });
  }
});

app.listen(port, () => {
  console.log(`✅ Server is running on port ${port}`);
});
