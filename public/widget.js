(function () {
  const iframe = document.createElement("iframe");
  iframe.src = window.__CHATBOT_URL__ || "https://shivaboys-chatbot.onrender.com/";
  iframe.style = "width: 350px; height: 450px; position: fixed; bottom: 20px; right: 20px; border:none; border-radius:10px; z-index:9999;";
  document.body.appendChild(iframe);
})();
