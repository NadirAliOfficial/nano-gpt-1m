"""
Web chat UI for the tiny Q&A model (model_qa.pt).
Zero external dependencies (Python http.server + PyTorch).

    python chat_app.py        # -> http://localhost:7861
"""

import http.server
import json
import os
import socketserver
import sys

import torch

from model import GPTConfig, NanoGPT
from tokenizer import ByteTokenizer

PORT = 7861
MODEL_PATH = "model_qa.pt"
PROMPT_Q = "### Question: "
PROMPT_A = "\n### Answer: "
END = "<|end|>"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
tokenizer = ByteTokenizer()
model = None


def try_load_model():
    global model
    if os.path.exists(MODEL_PATH) and model is None:
        try:
            ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
            m = NanoGPT(ckpt["config"]).to(device)
            m.load_state_dict(ckpt["model_state_dict"])
            m.eval()
            model = m
            print(f"Loaded {MODEL_PATH} ({m.count_parameters():,} params) on {device}")
        except Exception as e:
            print(f"Error loading model: {e}")


@torch.no_grad()
def ask(question, temperature=0.6, top_k=20, max_new_tokens=64):
    prompt = f"{PROMPT_Q}{question}{PROMPT_A}"
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    buf = bytearray()
    for tok in model.stream_generate(idx, max_new_tokens, temperature=temperature, top_k=top_k):
        buf.append(max(0, min(255, int(tok))))
        text = buf.decode("utf-8", errors="ignore")
        if END in text or "\n###" in text or "\n\n" in text:
            break
    ans = buf.decode("utf-8", errors="ignore")
    for stop in (END, "\n###", "\n\n", "\n"):
        if stop in ans:
            ans = ans.split(stop)[0]
    return ans.strip()


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NanoQA | Tiny Question Answering Model</title>
<style>
  :root {
    --bg: #0b0f17; --panel: #141a24; --border: rgba(255,255,255,0.08);
    --accent: #22c55e; --user: #2563eb; --text: #e7edf3; --muted: #93a1b0;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: radial-gradient(circle at top, #12203a, #0b0f17 55%); color: var(--text);
    min-height: 100vh; display: flex; flex-direction: column; align-items: center; }
  header { width: 100%; max-width: 720px; padding: 1.4rem 1rem 0.6rem; }
  h1 { font-size: 1.3rem; font-weight: 700; display: flex; align-items: center; gap: .5rem; }
  .badge { background: linear-gradient(135deg,#16a34a,#22c55e); font-size: .7rem;
    padding: .2rem .5rem; border-radius: 6px; letter-spacing: .04em; }
  .sub { color: var(--muted); font-size: .82rem; margin-top: .4rem; line-height: 1.5; }
  main { width: 100%; max-width: 720px; flex: 1; display: flex; flex-direction: column;
    padding: 1rem; gap: .8rem; }
  #chat { flex: 1; min-height: 55vh; background: var(--panel); border: 1px solid var(--border);
    border-radius: 14px; padding: 1rem; overflow-y: auto; display: flex; flex-direction: column; gap: .7rem; }
  .msg { max-width: 82%; padding: .6rem .85rem; border-radius: 12px; font-size: .95rem;
    line-height: 1.45; white-space: pre-wrap; word-wrap: break-word; }
  .msg.user { align-self: flex-end; background: var(--user); color: #fff; border-bottom-right-radius: 3px; }
  .msg.bot { align-self: flex-start; background: #1e2836; border: 1px solid var(--border);
    border-bottom-left-radius: 3px; font-family: "JetBrains Mono", ui-monospace, monospace; }
  .msg.bot.thinking { color: var(--muted); font-style: italic; }
  .chips { display: flex; flex-wrap: wrap; gap: .4rem; }
  .chip { background: rgba(255,255,255,.05); border: 1px solid var(--border); color: var(--muted);
    padding: .35rem .6rem; border-radius: 999px; font-size: .78rem; cursor: pointer; }
  .chip:hover { color: #fff; border-color: var(--accent); }
  .row { display: flex; gap: .6rem; }
  #q { flex: 1; background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
    padding: .8rem 1rem; color: #fff; font-size: 1rem; outline: none; }
  #q:focus { border-color: var(--accent); }
  #send { background: linear-gradient(135deg,#16a34a,#22c55e); border: none; color: #fff;
    font-weight: 600; padding: 0 1.3rem; border-radius: 12px; cursor: pointer; }
  #send:disabled { opacity: .5; cursor: not-allowed; }
  .note { color: var(--muted); font-size: .74rem; text-align: center; padding-bottom: .8rem; }
</style>
</head>
<body>
  <header>
    <h1><span class="badge">~5M PARAMS</span> NanoQA</h1>
    <div class="sub">A tiny byte-level GPT trained only on synthetic templates:
      small arithmetic (0-99), the times table, a fixed set of facts, opposites,
      word length, and greetings. Outside those, it makes things up.</div>
  </header>
  <main>
    <div id="chat">
      <div class="msg bot">Ask me a question. Try the suggestions below.</div>
    </div>
    <div class="chips" id="chips"></div>
    <div class="row">
      <input id="q" placeholder="What is 23 plus 19?" autocomplete="off">
      <button id="send" onclick="send()">Ask</button>
    </div>
    <div class="note">NanoQA runs locally on your machine. Answers are often wrong &mdash; that is expected at this size.</div>
  </main>
<script>
  const chat = document.getElementById('chat');
  const q = document.getElementById('q');
  const send_btn = document.getElementById('send');
  const samples = ["What is 23 plus 19?", "What is 84 minus 37?", "What is 6 times 7?",
    "What is the capital of Italy?", "What color is the sky?", "What is the opposite of up?",
    "How many days are in a week?", "Who are you?"];
  const chips = document.getElementById('chips');
  samples.forEach(s => { const c = document.createElement('div'); c.className = 'chip';
    c.textContent = s; c.onclick = () => { q.value = s; send(); }; chips.appendChild(c); });

  function add(text, cls) {
    const d = document.createElement('div'); d.className = 'msg ' + cls; d.textContent = text;
    chat.appendChild(d); chat.scrollTop = chat.scrollHeight; return d;
  }
  async function send() {
    const text = q.value.trim(); if (!text) return;
    add(text, 'user'); q.value = ''; send_btn.disabled = true;
    const pending = add('thinking...', 'bot thinking');
    try {
      const res = await fetch('/api/ask', { method: 'POST',
        headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ question: text }) });
      const data = await res.json();
      pending.classList.remove('thinking');
      pending.textContent = data.error ? ('[' + data.error + ']') : (data.answer || '(no answer)');
    } catch (e) {
      pending.classList.remove('thinking'); pending.textContent = '[network error: ' + e.message + ']';
    } finally { send_btn.disabled = false; q.focus(); }
  }
  q.addEventListener('keydown', e => { if (e.key === 'Enter') send(); });
  q.focus();
</script>
</body>
</html>
"""


class Handler(http.server.SimpleHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, HTML_PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path == "/api/status":
            try_load_model()
            self._send(200, json.dumps({"model_loaded": model is not None, "device": str(device)}).encode())
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != "/api/ask":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception:
            self._send(400, json.dumps({"error": "bad request"}).encode())
            return
        question = str(payload.get("question", "")).strip()
        try_load_model()
        if model is None:
            self._send(200, json.dumps({"error": "model_qa.pt not found. Run: python train_qa.py"}).encode())
            return
        if not question:
            self._send(200, json.dumps({"error": "empty question"}).encode())
            return
        answer = ask(question,
                     temperature=float(payload.get("temperature", 0.6)),
                     top_k=int(payload.get("top_k", 20)))
        self._send(200, json.dumps({"answer": answer}).encode())


def run(port=PORT):
    try_load_model()
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", port), Handler) as httpd:
        print(f"NanoQA chat UI running at http://localhost:{port}  (Ctrl+C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else PORT)
