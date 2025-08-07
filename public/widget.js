// ---------------------------
// Sage AI Widget JavaScript
// ---------------------------

// Store chat history for the current session
let conversationHistory = [];

// Function to append messages to the chat UI
function appendMessage(sender, text) {
  const messagesContainer = document.getElementById("chat-messages");

  const messageElem = document.createElement("div");
  messageElem.classList.add("message", sender);
  messageElem.innerHTML = text;

  messagesContainer.appendChild(messageElem);
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
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
        history: conversationHistory // Send history to the backend
      }),
    });

    const data = await response.json();
    const answer = data.answer || "Sorry, I couldn't get an answer.";

    // Display Sage's reply
    appendMessage("bot", answer);

    // Add Sage's reply to conversation history
    conversationHistory.push({ role: "assistant", content: answer });

    // Keep history short-term (last 10 exchanges)
    if (conversationHistory.length > 20) {
      conversationHistory = conversationHistory.slice(-20);
    }
  } catch (error) {
    console.error("Error sending message:", error);
    appendMessage("bot", "⚠️ There was a problem connecting to Sage. Please try again.");
  }
}

// Event listeners
document.getElementById("chat-send").addEventListener("click", sendMessage);
document.getElementById("chat-input").addEventListener("keypress", function (e) {
  if (e.key === "Enter") {
    sendMessage();
  }
});
