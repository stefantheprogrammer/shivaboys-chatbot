// widget.js

// Create a floating chat button in the bottom right
const chatButton = document.createElement('button');
chatButton.textContent = "Chat";
chatButton.style.position = "fixed";
chatButton.style.bottom = "20px";
chatButton.style.right = "20px";
chatButton.style.zIndex = "1000";
chatButton.style.padding = "10px 20px";
chatButton.style.backgroundColor = "#003366"; // your blue
chatButton.style.color = "white";
chatButton.style.borderRadius = "8px";
chatButton.style.border = "none";
chatButton.style.cursor = "pointer";
document.body.appendChild(chatButton);

chatButton.onclick = () => {
  alert("Open chatbot UI here");
};
