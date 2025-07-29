(function () {
  const iframe = document.createElement("iframe");
  iframe.src = "https://shivaboys-chatbot.onrender.com/widget.html";
  iframe.style = `
    width: 350px;
    height: 450px;
    position: fixed;
    bottom: 20px;
    right: 20px;
    border: none;
    border-radius: 10px;
    z-index: 9999;
    box-shadow: 0 0 15px rgba(0, 0, 0, 0.2);
  `;
  document.body.appendChild(iframe);
})();

