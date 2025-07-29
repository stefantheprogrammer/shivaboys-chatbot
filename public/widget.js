(function () {
  const iframe = document.createElement("iframe");
  iframe.src = window.__CHATBOT_URL__ || "https://shivaboys-chatbot.onrender.com/widget.html";
  iframe.style.width = "350px";
  iframe.style.height = "450px";
  iframe.style.position = "fixed";
  iframe.style.bottom = "20px";
  iframe.style.right = "20px";
  iframe.style.border = "none";
  iframe.style.borderRadius = "10px";
  iframe.style.zIndex = "9999";
  document.body.appendChild(iframe);
})();
