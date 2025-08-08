// ---------------------------
// Sage AI Widget JavaScript
// ---------------------------

// Store chat history for the current session
let conversationHistory = [];

// Get chat messages container once
const chatMessages = document.getElementById("chat-messages");

// Function to append messages to the chat UI
function appendMessage(role, text) {
  const msg = document.createElement("div");
  msg.className = `message ${role}`;

  // Normalize newlines:
  // 1. Trim leading/trailing whitespace
  // 2. Replace multiple consecutive newlines with one
  // 3. Replace single newlines with <br> for line breaks
  let formatted = text.trim()
    .replace(/(\r\n|\r|\n){2,}/g, "<br>")   // multiple newlines to one <br>
    .replace(/(\r\n|\r|\n)/g, "<br>")        // single newline to <br>
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>") // bold markdown
    .replace(/`(.*?)`/g, "<code>$1</code>");          // inline code

  msg.innerHTML = formatted;
  chatMessages.appendChild(msg);

  // Scroll to bottom after adding message
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Function to send a message to the server
async function sendMessage() {
  const inputElem = document.getElementById("chat-input");
  const query = inputElem.value.trim();

  if (!query) return;

  // Display user's message
  appendMessage("user", query);

  // Add user message to conversation history
  conversationHistory.push({ role: "user", content: query });

  inputElem.value = "";

  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        query,
        history: conversationHistory, // Send history to the backend
      }),
    });

    const data = await response.json();
    const answer = data.answer || "Sorry, I couldn't get an answer.";

    // Display Sage's reply
    appendMessage("bot", answer);

    // Add Sage's reply to conversation history
    conversationHistory.push({ role: "assistant", content: answer });

    // Keep history short-term (last 20 messages total)
    if (conversationHistory.length > 20) {
      conversationHistory = conversationHistory.slice(-20);
    }
  } catch (error) {
    console.error("Error sending message:", error);
    appendMessage(
      "bot",
      "⚠️ There was a problem connecting to Sage. Please try again."
    );
  }
}

// Event listeners
document.getElementById("chat-send").addEventListener("click", sendMessage);
document.getElementById("chat-input").addEventListener("keypress", function (e) {
  if (e.key === "Enter") {
    sendMessage();
  }
});
