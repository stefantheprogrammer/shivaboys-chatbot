// Make sure you include marked.js in your HTML:
// <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>

const messagesEl = document.getElementById('messages');
const inputText = document.getElementById('input-text');
const sendBtn = document.getElementById('send-btn');

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

  // Parse markdown text to HTML using marked
  msg.innerHTML = marked.parse(text);

  if (isTemporary) {
    msg.dataset.temp = 'true';
  }
  messagesEl.appendChild(msg);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function updateLastBotMessage(text) {
  const lastMsg = messagesEl.querySelector('.message.bot[data-temp="true"]');
  if (lastMsg) {
    lastMsg.innerHTML = marked.parse(text);
    lastMsg.removeAttribute('data-temp');
  }
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

async function sendMessageToBackend(text) {
  const res = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message: text }),
  });

  if (!res.ok) {
    throw new Error('Network response was not ok');
  }
  const data = await res.json();
  return data.reply;
}
