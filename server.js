const express = require("express");
const path = require("path");
require("dotenv").config();
const OpenAI = require("openai");

const app = express();
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

app.use(express.static(path.join(__dirname, "public")));
app.use(express.json());

// Serve widget.html on root URL
app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "widget.html"));
});

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

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));