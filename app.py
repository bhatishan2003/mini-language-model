"""
app.py
======
Step 5: Flask web application exposing the chat interface.

Run:
    python app.py
    # → open http://localhost:5000 in your browser

Environment variables (optional):
    VOCAB_PATH  – path to vocab.json   (default: data/vocab.json)
    CKPT_PATH   – path to checkpoint   (default: checkpoints/finetune_best.pt)
    PORT        – server port          (default: 5000)
"""

import os
from flask import Flask, request, jsonify, render_template_string

from inference import ChatEngine

# ---------------------------------------------------------------------------
# Initialise
# ---------------------------------------------------------------------------

app = Flask(__name__)

VOCAB_PATH = os.environ.get("VOCAB_PATH", "data/vocab.json")
CKPT_PATH = os.environ.get("CKPT_PATH", "checkpoints/finetune_best.pt")

engine: ChatEngine | None = None


def get_engine() -> ChatEngine:
    global engine
    if engine is None:
        engine = ChatEngine(
            vocab_path=VOCAB_PATH,
            ckpt_path=CKPT_PATH,
        )
    return engine


# ---------------------------------------------------------------------------
# HTML template (inline – no separate template folder needed)
# ---------------------------------------------------------------------------

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MiniGPT – Chat</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;700;800&display=swap');

  :root {
    --bg:      #0a0a0f;
    --surface: #111118;
    --border:  #1e1e2e;
    --accent:  #7c6af7;
    --accent2: #a78bfa;
    --text:    #e2e0ff;
    --muted:   #6b6990;
    --user-bg: #1a1830;
    --bot-bg:  #131320;
  }

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'Syne', sans-serif;
    background: var(--bg);
    color: var(--text);
    height: 100dvh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  /* ── Header ── */
  header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 16px 24px;
    border-bottom: 1px solid var(--border);
    background: var(--surface);
    flex-shrink: 0;
  }
  .logo {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-weight: 800; font-size: 15px; color: #fff;
    flex-shrink: 0;
  }
  header h1 { font-size: 18px; font-weight: 800; letter-spacing: -0.3px; }
  header span {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--muted);
    margin-left: 4px;
    background: var(--border);
    padding: 2px 8px;
    border-radius: 20px;
  }

  /* ── Chat window ── */
  #chat {
    flex: 1;
    overflow-y: auto;
    padding: 24px;
    display: flex;
    flex-direction: column;
    gap: 16px;
    scroll-behavior: smooth;
  }
  #chat::-webkit-scrollbar { width: 4px; }
  #chat::-webkit-scrollbar-track { background: transparent; }
  #chat::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }

  /* ── Messages ── */
  .msg {
    display: flex;
    gap: 12px;
    max-width: 780px;
    animation: fadeUp 0.3s ease both;
  }
  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  .msg.user  { align-self: flex-end;  flex-direction: row-reverse; }
  .msg.bot   { align-self: flex-start; }

  .avatar {
    width: 32px; height: 32px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; flex-shrink: 0;
    margin-top: 2px;
  }
  .user .avatar { background: var(--accent); }
  .bot  .avatar { background: var(--border); border: 1px solid var(--border); }

  .bubble {
    padding: 12px 16px;
    border-radius: 16px;
    font-size: 15px;
    line-height: 1.65;
    max-width: min(580px, calc(100vw - 80px));
    word-break: break-word;
  }
  .user .bubble {
    background: var(--user-bg);
    border: 1px solid var(--accent);
    border-bottom-right-radius: 4px;
    color: var(--text);
  }
  .bot .bubble {
    background: var(--bot-bg);
    border: 1px solid var(--border);
    border-bottom-left-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13.5px;
    color: #d0ceee;
  }

  /* ── Typing indicator ── */
  .typing { display: flex; gap: 5px; align-items: center; padding: 14px 16px; }
  .typing span {
    width: 7px; height: 7px;
    background: var(--muted);
    border-radius: 50%;
    animation: blink 1.2s infinite;
  }
  .typing span:nth-child(2) { animation-delay: 0.2s; }
  .typing span:nth-child(3) { animation-delay: 0.4s; }
  @keyframes blink { 0%,80%,100% { opacity: .2 } 40% { opacity: 1 } }

  /* ── Input bar ── */
  #input-bar {
    display: flex;
    gap: 10px;
    padding: 16px 24px;
    border-top: 1px solid var(--border);
    background: var(--surface);
    flex-shrink: 0;
  }
  #msg-input {
    flex: 1;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 12px 16px;
    color: var(--text);
    font-family: 'Syne', sans-serif;
    font-size: 15px;
    outline: none;
    transition: border-color 0.2s;
    resize: none;
    min-height: 48px;
    max-height: 140px;
    line-height: 1.5;
  }
  #msg-input:focus { border-color: var(--accent); }
  #msg-input::placeholder { color: var(--muted); }

  #send-btn {
    width: 48px; height: 48px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    border: none; border-radius: 12px;
    cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    transition: opacity 0.2s, transform 0.1s;
    flex-shrink: 0;
    align-self: flex-end;
  }
  #send-btn:hover   { opacity: 0.88; }
  #send-btn:active  { transform: scale(0.93); }
  #send-btn:disabled { opacity: 0.35; cursor: not-allowed; }
  #send-btn svg { width: 20px; height: 20px; fill: none; stroke: #fff; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }

  /* ── System message ── */
  .system-msg {
    text-align: center;
    font-size: 12px;
    color: var(--muted);
    font-family: 'JetBrains Mono', monospace;
    padding: 4px 0 8px;
  }
</style>
</head>
<body>

<header>
  <div class="logo">μG</div>
  <h1>MiniGPT</h1>
  <span>GRU · v1.0</span>
</header>

<div id="chat">
  <div class="system-msg">── conversation started ──</div>
  <div class="msg bot">
    <div class="avatar">🤖</div>
    <div class="bubble">Hello! I'm MiniGPT — a small GRU-based language model for stories.</div>
  </div>
</div>

<div id="input-bar">
  <textarea id="msg-input" placeholder="Ask me anything…" rows="1" autofocus></textarea>
  <button id="send-btn" title="Send">
    <svg viewBox="0 0 24 24"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
  </button>
</div>

<script>
const chat     = document.getElementById('chat');
const input    = document.getElementById('msg-input');
const sendBtn  = document.getElementById('send-btn');

function addMessage(role, text) {
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  div.innerHTML = `
    <div class="avatar">${role === 'user' ? '🧑' : '🤖'}</div>
    <div class="bubble">${escHtml(text)}</div>`;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

function addTyping() {
  const div = document.createElement('div');
  div.className = 'msg bot';
  div.id = 'typing';
  div.innerHTML = `<div class="avatar">🤖</div><div class="bubble typing"><span></span><span></span><span></span></div>`;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

function removeTyping() {
  const el = document.getElementById('typing');
  if (el) el.remove();
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\n/g,'<br>');
}

async function sendMessage() {
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  autoResize();

  addMessage('user', text);
  sendBtn.disabled = true;
  addTyping();

  try {
    const res  = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text }),
    });
    const data = await res.json();
    removeTyping();
    addMessage('bot', data.reply || 'No response.');
  } catch (err) {
    removeTyping();
    addMessage('bot', '⚠ Connection error. Is the server running?');
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
}

// Auto-resize textarea
function autoResize() {
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 140) + 'px';
}

input.addEventListener('input', autoResize);
input.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});
sendBtn.addEventListener('click', sendMessage);
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"reply": "Please send a non-empty message."})
    try:
        reply = get_engine().respond(message)
    except Exception as e:
        app.logger.error(f"Inference error: {e}")
        reply = "Sorry, I encountered an error. Please try again."
    return jsonify({"reply": reply})


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n🌐  MiniGPT web UI → http://localhost:{port}\n")
    # Pre-load model before first request
    get_engine()
    app.run(host="0.0.0.0", port=port, debug=False)
