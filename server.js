import express from "express";
import cors from "cors";
import { HuggingFaceInferenceEmbeddings } from "langchain/embeddings/hf";
import { ChatHuggingFaceInference } from "langchain/chat_models/hf";
import { RetrievalQAChain } from "langchain/chains";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { PromptTemplate } from "langchain/prompts";

const app = express();
app.use(cors());
app.use(express.json());

const hfKey = process.env.HUGGINGFACEHUB_API_KEY;
if (!hfKey) {
  throw new Error("Missing HUGGINGFACEHUB_API_KEY environment variable");
}

const embeddings = new HuggingFaceInferenceEmbeddings({ apiKey: hfKey });
const chatModel = new ChatHuggingFaceInference({
  model: "HuggingFaceH4/zephyr-7b-beta",
  apiKey: hfKey,
  temperature: 0.6
});

const prompt = PromptTemplate.fromTemplate(`
You are an assistant for a school website. Answer based on the context below.

Context:
{context}

Question: {question}
`);

const loader = async () => {
  // Hardcoded documents — later you can load from a folder or site
  const documents = [
    {
      title: "Welcome",
      content: "Shiva Boys’ Hindu College is a prestigious institution located in Trinidad and Tobago."
    },
    {
      title: "Vision",
      content: "To produce disciplined, productive and respected citizens of Trinidad and Tobago."
    }
  ];

  const formattedDocs = documents.map(doc => ({
    pageContent: doc.title + "\n\n" + doc.content
  }));

  const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 500, chunkOverlap: 50 });
  const docs = await splitter.splitDocuments(formattedDocs);

  const vectorStore = await MemoryVectorStore.fromDocuments(docs, embeddings);
  return vectorStore;
};

const vectorStorePromise = loader();

app.post("/chat", async (req, res) => {
  const { message } = req.body;
  if (!message) return res.status(400).json({ error: "No message provided" });

  try {
    const vectorStore = await vectorStorePromise;

    const chain = RetrievalQAChain.fromLLM(chatModel, vectorStore.asRetriever(), {
      prompt,
      returnSourceDocuments: true
    });

    const response = await chain.call({ query: message });
    res.json({ response: response.text });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to process request" });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Chatbot running on port ${PORT}`));
