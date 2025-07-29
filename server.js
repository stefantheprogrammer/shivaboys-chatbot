const express = require("express");
const path = require("path");
require("dotenv").config();
const OpenAI = require("openai"); // Updated for SDK v4+

const app = express();

// ✅ Initialize OpenAI SDK (v4+)
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

app.use(express.json());

// ✅ Serve all static files from the public directory
app.use(express.static(path.join(__dirname, "public")));

// ✅ Chat endpoint
app.post("/chat", async (req, res) => {
  try {
    const userMessage = req.body.message;

    const chatResponse = await openai.chat.completions.create({
      model: "gpt-4",
      messages: [{ role: "user", content: userMessage }],
    });

    res.json({ reply: chatResponse.choices[0].message.content });
  } catch (err) {
    console.error("OpenAI error:", err);
    res.status(500).json({ reply: "Sorry, something went wrong." });
  }
});

// ✅ Start the server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`✅ Server running on port ${PORT}`));
