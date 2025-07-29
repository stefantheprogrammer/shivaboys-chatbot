const express = require("express");
const cors = require("cors");
const path = require("path");
require("dotenv").config();
const { OpenAI } = require("openai");

const app = express();
const port = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());
app.use(express.static("public"));

app.get("/widget.html", (req, res) => {
  res.setHeader("Content-Type", "text/html");
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.sendFile(path.join(__dirname, "public", "widget.html"));
});

app.get("/widget.js", (req, res) => {
  res.setHeader("Content-Type", "application/javascript");
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.sendFile(path.join(__dirname, "public", "widget.js"));
});

app.post("/chat", async (req, res) => {
  try {
    const openai = new OpenAI({
      apiKey: process.env.GROQ_API_KEY,
      baseURL: "https://api.groq.com/openai/v1",
    });

    const completion = await openai.chat.completions.create({
      model: "llama3-8b-8192",
      messages: [
        {
          role: "system",
          content:
            "You are a helpful AI assistant for Shiva Boys' Hindu College in Trinidad and Tobago. Answer clearly and politely.",
        },
        { role: "user", content: req.body.message },
      ],
    });

    const reply = completion.choices[0].message.content;
    res.json({ reply });
  } catch (error) {
    console.error("Groq API error:", error);
    res.status(500).json({
      error: "Sorry, something went wrong processing your request.",
    });
  }
});

app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});
