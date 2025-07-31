
document.addEventListener("DOMContentLoaded", () => {
  const input = document.getElementById("user-input");
  const button = document.getElementById("send-button");
  const messages = document.getElementById("chat-messages");
  const minimize = document.getElementById("chat-minimize");
  const widget = document.getElementById("chat-widget");

  minimize.addEventListener("click", () => {
    widget.classList.toggle("minimized");
  });

  button.addEventListener("click", () => {
    const userText = input.value.trim();
    if (!userText) return;
    addMessage("user", userText);
    fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: userText })
    })
    .then(res => res.json())
    .then(data => {
      addMessage("bot", data.response);
    });
    input.value = "";
  });

  function addMessage(sender, text) {
    const msg = document.createElement("div");
    msg.classList.add("message", sender);
    msg.textContent = text;
    messages.appendChild(msg);
    messages.scrollTop = messages.scrollHeight;
  }
});
