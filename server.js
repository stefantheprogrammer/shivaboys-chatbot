const express = require("express");
const path = require("path");
require("dotenv").config();
const OpenAI = require("openai");

const app = express();
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// Serve static files from "public"
app.use(express.static(path.join(__dirname, "public")));
app.use(express.json());
