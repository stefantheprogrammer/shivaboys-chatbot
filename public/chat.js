const messagesEl = document.getElementById('messages');
const inputText = document.getElementById('input-text');
const sendBtn = document.getElementById('send-btn');

let sessionId = localStorage.getItem('chatSessionId') || null;

inputText.addEventListener('input', () => {
  sendBtn.disabled = inputText.value.trim() === '';
});

sendBtn.addEventListener('click', async () => {
  const text = inputText.value.trim();
  if (!text) return;

  addMessage(text, 'user');
  inputText.value = '';
  sendBtn.disabled = true;

  addMessage('Thinking...', 'bot', true);

  try {
    const reply = await sendMessageToBackend(text);
    updateLastBotMessage(reply);
  } catch (e) {
    updateLastBotMessage("Sorry, something went wrong.");
    console.error(e);
  }
});

function addMessage(text, sender, isTemporary = false) {
  const msg = document.createElement('div');
  msg.classList.add('message', sender);

  // Markdown-ish simple formatting (bold, line breaks)
  const formatted = text
    .replace(/(?:\r\n|\r|\n)/g, "<br>")
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/`(.*?)`/g, "<code>$1</code>");

  msg.innerHTML = formatted;

  if (isTemporary) {
    msg.dataset.temp = 'true';
  }

  messagesEl.appendChild(msg);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function updateLastBotMessage(text) {
  const lastMsg = messagesEl.querySelector('.message.bot[data-temp="true"]');
  if (lastMsg) {
    // Update with formatted markdown
    const formatted = text
      .replace(/(?:\r\n|\r|\n)/g, "<br>")
      .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
      .replace(/`(.*?)`/g, "<code>$1</code>");
    lastMsg.innerHTML = formatted;
    lastMsg.removeAttribute('data-temp');
  }
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

async function sendMessageToBackend(text) {
  const payload = { message: text };
  if (sessionId) {
    payload.sessionId = sessionId;
  }

  const res = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    throw new Error('Network response was not ok');
  }

  const data = await res.json();

  // Save sessionId if received from backend
  if (data.sessionId && data.sessionId !== sessionId) {
    sessionId = data.sessionId;
    localStorage.setItem('chatSessionId', sessionId);
  }

  return data.reply;
}
