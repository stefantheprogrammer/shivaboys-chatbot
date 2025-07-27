
const chatButton = document.getElementById('chatbot-button');
const chatContainer = document.getElementById('chat-container');
const chatMessages = document.getElementById('chat-messages');
const userInput = document.getElementById('user-input');
const micButton = document.getElementById('mic-button');

chatButton.onclick = () => {
  chatContainer.style.display = chatContainer.style.display === 'flex' ? 'none' : 'flex';
  chatContainer.style.flexDirection = 'column';
};

async function sendMessage() {
  const message = userInput.value;
  if (!message) return;
  appendMessage('You', message);
  userInput.value = '';
  const res = await fetch('/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message })
  });
  const data = await res.json();
  appendMessage('SBHC AI', data.reply);
  speak(data.reply);
}

function appendMessage(sender, text) {
  const msg = document.createElement('div');
  msg.innerHTML = `<strong>${sender}:</strong> ${text}`;
  chatMessages.appendChild(msg);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function speak(text) {
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = 'en-US';
  speechSynthesis.speak(utterance);
}

micButton.onclick = () => {
  const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
  recognition.lang = 'en-US';
  recognition.start();
  recognition.onresult = event => {
    userInput.value = event.results[0][0].transcript;
    sendMessage();
  };
};
