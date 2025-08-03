import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/hf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { ChatGroq } from "@langchain/groq";
import { RetrievalQAChain } from "langchain/chains";
import { BufferMemory } from "langchain/memory";
import { Document } from "@langchain/core/documents";

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
      content: "Shiva Boys' Hindu College is a government-assisted secondary school located in Trinidad and Tobago..."
    },
    {
      title: "Curriculum", 
      content: "The curriculum includes Mathematics, English, Science, Information Technology, Business Studies, and Modern Languages..."
    },
    // Add more documents as needed
  ];

  // Convert to Document objects
  const formattedDocs = docs.map(doc => new Document({
    pageContent: doc.content,
    metadata: { title: doc.title }
  }));

  const splitter = new RecursiveCharacterTextSplitter({ 
    chunkSize: 500, 
    chunkOverlap: 50 
  });
  
  const splitDocs = await splitter.splitDocuments(formattedDocs);

  const embeddings = new HuggingFaceTransformersEmbeddings({
    modelName: "Xenova/all-MiniLM-L6-v2"
  });

  vectorStore = await MemoryVectorStore.fromDocuments(splitDocs, embeddings);
  console.log("✅ Vector store loaded successfully");
}

// Initialize on startup
loadDocs().catch(console.error);

app.post("/chat", async (req, res) => {
  const { message } = req.body;

  if (!vectorStore) {
    return res.status(500).json({ error: "Vector store not initialized" });
  }

  try {
    const model = new ChatGroq({
      apiKey: process.env.GROQ_API_KEY,
      model: "llama3-8b-8192",
      temperature: 0.7,
    });

    const chain = RetrievalQAChain.fromLLM(
      model, 
      vectorStore.asRetriever(),
      {
        returnSourceDocuments: true,
      }
    );

    const response = await chain.call({ query: message });

    res.json({ 
      response: response.text,
      sources: response.sourceDocuments?.map(doc => doc.metadata.title) || []
    });
  } catch (error) {import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { ChatGroq } from "@langchain/groq";
import { Document } from "@langchain/core/documents";

dotenv.config();

const app = express();
const port = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

let documents = [];

// Simple text similarity function (no external API needed)
function cosineSimilarity(a, b) {
  const wordsA = a.toLowerCase().split(/\W+/).filter(w => w.length > 2);
  const wordsB = b.toLowerCase().split(/\W+/).filter(w => w.length > 2);
  
  const allWords = [...new Set([...wordsA, ...wordsB])];
  
  const vectorA = allWords.map(word => wordsA.filter(w => w === word).length);
  const vectorB = allWords.map(word => wordsB.filter(w => w === word).length);
  
  const dotProduct = vectorA.reduce((sum, a, i) => sum + a * vectorB[i], 0);
  const magnitudeA = Math.sqrt(vectorA.reduce((sum, a) => sum + a * a, 0));
  const magnitudeB = Math.sqrt(vectorB.reduce((sum, b) => sum + b * b, 0));
  
  return magnitudeA && magnitudeB ? dotProduct / (magnitudeA * magnitudeB) : 0;
}

// Find most relevant documents
function findRelevantDocs(query, topK = 3) {
  const scores = documents.map(doc => ({
    doc,
    score: cosineSimilarity(query, doc.pageContent)
  }));
  
  return scores
    .sort((a, b) => b.score - a.score)
    .slice(0, topK)
    .map(item => item.doc);
}

// Load docs into memory
async function loadDocs() {
  const docs = [
    {
      title: "Welcome",
      content: "Shiva Boys' Hindu College is a government-assisted secondary school located in Penal, Trinidad and Tobago. Founded in 1941, the school has a rich history of academic excellence and cultural diversity. We serve students from Forms 1-6, preparing them for CXC and CAPE examinations. Our motto is 'Knowledge is Light' and we strive to provide quality education in a supportive environment."
    },
    {
      title: "Curriculum and Subjects", 
      content: "The curriculum includes core subjects: Mathematics, English Language, English Literature, Science (Biology, Chemistry, Physics), Information Technology, Business Studies, Modern Languages (Spanish, French), Social Studies, Geography, History, Physical Education, and Visual Arts. We offer both academic and technical subjects to prepare students for various career paths. Advanced level subjects are available for Forms 6 and 7."
    },
    {
      title: "Admissions and Enrollment",
      content: "Admission to Form 1 is through the Secondary Entrance Assessment (SEA) conducted by the Ministry of Education. Students are selected based on their SEA results and school preferences. The school accepts approximately 180 new students each year across 6 classes. We welcome students from all backgrounds and communities in Trinidad and Tobago."
    },
    {
      title: "School Hours and Contact",
      content: "Shiva Boys' Hindu College is located in Penal, Trinidad and Tobago. Regular school hours are 8:00 AM to 3:00 PM, Monday to Friday. The school office is open from 7:30 AM to 4:00 PM. For inquiries, please contact the school office during regular school hours. We also have after-school programs and extra-curricular activities."
    },
    {
      title: "Facilities and Resources",
      content: "Our school features modern classrooms, science laboratories, computer labs, library, sports facilities including basketball and football courts, and a multipurpose hall. We have qualified teachers and support staff dedicated to student success. The school also provides counseling services and career guidance."
    },
    {
      title: "Extra-curricular Activities",
      content: "Students can participate in various clubs and societies including Drama Club, Debate Society, Science Club, Mathematics Club, Cultural groups, and Sports teams. We regularly participate in inter-school competitions and cultural events. These activities help develop leadership skills and talents beyond academics."
    }
  ];

  try {
    // Convert to Document objects
    const formattedDocs = docs.map(doc => new Document({
      pageContent: doc.content,
      metadata: { title: doc.title }
    }));

    const splitter = new RecursiveCharacterTextSplitter({ 
      chunkSize: 600, 
      chunkOverlap: 100 
    });
    
    documents = await splitter.splitDocuments(formattedDocs);
    console.log(`✅ Loaded ${documents.length} document chunks`);
  } catch (error) {
    console.error("❌ Error loading docs:", error);
  }
}

// Initialize on startup
loadDocs();

app.post("/chat", async (req, res) => {
  const { message } = req.body;

  if (!message) {
    return res.status(400).json({ error: "Message is required" });
  }

  if (documents.length === 0) {
    return res.status(500).json({ error: "Documents not loaded" });
  }

  try {
    // Find relevant documents
    const relevantDocs = findRelevantDocs(message, 3);
    const context = relevantDocs.map(doc => doc.pageContent).join('\n\n');
    
    const model = new ChatGroq({
      apiKey: process.env.GROQ_API_KEY,
      model: "llama3-8b-8192",
      temperature: 0.7,
    });

    // Create a prompt with context
    const prompt = `You are a helpful assistant for Shiva Boys' Hindu College in Trinidad and Tobago. Use the following context to answer the student's question. If the context doesn't contain relevant information, politely say you don't have that specific information and suggest they contact the school office.

Context:
${context}

Question: ${message}

Answer:`;

    const response = await model.invoke(prompt);

    res.json({ 
      response: response.content,
      sources: relevantDocs.map(doc => doc.metadata.title)
    });
  } catch (error) {
    console.error("Chat error:", error);
    res.status(500).json({ 
      error: "Failed to generate response",
      details: error.message 
    });
  }
});

// Health check endpoint
app.get("/health", (req, res) => {
  res.json({ 
    status: "OK", 
    documentsLoaded: documents.length,
    timestamp: new Date().toISOString()
  });
});

// Root endpoint
app.get("/", (req, res) => {
  res.json({ message: "Shiva Boys' Hindu College Chatbot API - Free Version" });
});

app.listen(port, () => {
  console.log(`✅ Server is running on port ${port}`);
});

    console.error("Chat error:", error);
    res.status(500).json({ error: "Failed to generate response" });
  }
});

// Health check endpoint
app.get("/health", (req, res) => {
  res.json({ status: "OK", vectorStore: !!vectorStore });
});

app.listen(port, () => {
  console.log(`✅ Server is running on port ${port}`);
});