const express = require('express');
const path = require('path');
const OpenAI = require('openai');

const app = express();
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

app.use(express.static('public'));
app.use(express.json());

app.post('/chat', async (req, res) => {
  try {
    const userMessage = req.body.message;

    const chatResponse = await openai.chat.completions.create({
      model: 'gpt-4',
      messages: [{ role: 'user', content: userMessage }],
    });

    const reply = chatResponse.choices[0].message.content;
    res.json({ reply });
  } catch (error) {
    console.error('OpenAI error:', error);
    res.status(500).json({ reply: 'Sorry, something went wrong.' });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
