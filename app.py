"""
NanoGPT-1M Web Playground.
Zero external dependencies (uses Python built-in http.server and PyTorch).
Serves a sleek, dark-mode web interface to test and chat with your 1M parameter model.
"""

import http.server
import json
import os
import socketserver
import sys
import threading
import time
import urllib.parse
import torch

from model import GPTConfig, NanoGPT
from tokenizer import ByteTokenizer

PORT = 7860
MODEL_PATH = "model_1m.pt"

# Global model state
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
tokenizer = ByteTokenizer()
model = None


def get_device_info():
    if torch.backends.mps.is_available():
        return "Apple Silicon (MPS Hardware Acceleration)"
    elif torch.cuda.is_available():
        return "NVIDIA CUDA Acceleration"
    return "CPU"


def try_load_model():
    global model
    if os.path.exists(MODEL_PATH) and model is None:
        print(f"Loading checkpoint '{MODEL_PATH}' onto {device}...")
        try:
            ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
            config = ckpt["config"]
            m = NanoGPT(config).to(device)
            m.load_state_dict(ckpt["model_state_dict"])
            m.eval()
            model = m
            print("Model loaded successfully into Web UI!")
        except Exception as e:
            print(f"Error loading model: {e}")


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>NanoGPT-1M Playground | Apple Silicon</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-primary: #0a0d14;
      --bg-card: rgba(22, 27, 34, 0.75);
      --border: rgba(255, 255, 255, 0.1);
      --accent: #8b5cf6;
      --accent-glow: rgba(139, 92, 246, 0.35);
      --text-main: #f3f4f6;
      --text-muted: #9ca3af;
      --apple-chip: #3b82f6;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
      background: radial-gradient(circle at top right, #1e1035, #0a0d14 60%);
      color: var(--text-main);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }

    header {
      padding: 1.5rem 2.5rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--border);
      backdrop-filter: blur(12px);
      background: rgba(10, 13, 20, 0.6);
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 0.8rem;
    }

    .brand-badge {
      background: linear-gradient(135deg, #ec4899, #8b5cf6);
      font-weight: 700;
      font-size: 0.8rem;
      padding: 0.3rem 0.6rem;
      border-radius: 6px;
      letter-spacing: 0.05em;
    }

    .hardware-pill {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      background: rgba(59, 130, 246, 0.15);
      border: 1px solid rgba(59, 130, 246, 0.4);
      padding: 0.4rem 0.9rem;
      border-radius: 9999px;
      font-size: 0.85rem;
      color: #93c5fd;
    }

    .pulse-dot {
      width: 8px;
      height: 8px;
      background: #10b981;
      border-radius: 50%;
      box-shadow: 0 0 10px #10b981;
      animation: pulse 2s infinite;
    }
    @keyframes pulse {
      0%, 100% { transform: scale(1); opacity: 1; }
      50% { transform: scale(1.3); opacity: 0.6; }
    }

    main {
      flex: 1;
      max-width: 1200px;
      width: 100%;
      margin: 0 auto;
      padding: 2rem;
      display: grid;
      grid-template-columns: 340px 1fr;
      gap: 1.5rem;
    }

    .card {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 1.5rem;
      backdrop-filter: blur(16px);
      box-shadow: 0 12px 30px rgba(0, 0, 0, 0.4);
    }

    .sidebar h2, .playground h2 {
      font-size: 1.15rem;
      font-weight: 600;
      margin-bottom: 1.2rem;
      display: flex;
      align-items: center;
      gap: 0.5rem;
      color: #e5e7eb;
    }

    .spec-item {
      display: flex;
      justify-content: space-between;
      padding: 0.65rem 0;
      border-bottom: 1px solid rgba(255, 255, 255, 0.06);
      font-size: 0.9rem;
    }
    .spec-label { color: var(--text-muted); }
    .spec-value { font-family: 'JetBrains Mono', monospace; font-weight: 600; color: #a78bfa; }

    .control-group {
      margin-top: 1.2rem;
    }
    .control-label {
      display: flex;
      justify-content: space-between;
      font-size: 0.85rem;
      color: var(--text-muted);
      margin-bottom: 0.4rem;
    }
    input[type=range] {
      width: 100%;
      accent-color: var(--accent);
      cursor: pointer;
    }

    .presets {
      display: flex;
      flex-wrap: wrap;
      gap: 0.4rem;
      margin-top: 0.8rem;
    }
    .preset-btn {
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid var(--border);
      color: var(--text-muted);
      padding: 0.35rem 0.65rem;
      border-radius: 8px;
      font-size: 0.78rem;
      cursor: pointer;
      transition: all 0.2s;
    }
    .preset-btn:hover {
      background: var(--accent-glow);
      color: #fff;
      border-color: var(--accent);
    }

    .playground {
      display: flex;
      flex-direction: column;
      gap: 1rem;
    }

    .chat-output {
      flex: 1;
      min-height: 380px;
      background: rgba(13, 17, 23, 0.85);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.2rem;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.95rem;
      line-height: 1.6;
      white-space: pre-wrap;
      overflow-y: auto;
      color: #e6edf3;
      box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.5);
    }

    .input-row {
      display: flex;
      gap: 0.8rem;
    }

    .prompt-input {
      flex: 1;
      background: rgba(13, 17, 23, 0.85);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 0.85rem 1.2rem;
      font-size: 1rem;
      font-family: inherit;
      color: #fff;
      outline: none;
      transition: border-color 0.2s;
    }
    .prompt-input:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px var(--accent-glow);
    }

    .generate-btn {
      background: linear-gradient(135deg, #7c3aed, #6366f1);
      border: none;
      color: white;
      font-weight: 600;
      padding: 0 1.6rem;
      border-radius: 12px;
      font-size: 0.95rem;
      cursor: pointer;
      box-shadow: 0 4px 15px var(--accent-glow);
      transition: transform 0.15s, opacity 0.2s;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    .generate-btn:hover {
      transform: translateY(-1px);
    }
    .generate-btn:active {
      transform: translateY(1px);
    }
    .generate-btn:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }

    .status-bar {
      display: flex;
      justify-content: space-between;
      font-size: 0.8rem;
      color: var(--text-muted);
      margin-top: 0.2rem;
    }

    @media (max-width: 860px) {
      main { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <span class="brand-badge">1M PARAMS</span>
      <h1>NanoGPT-1M Studio</h1>
    </div>
    <div class="hardware-pill">
      <span class="pulse-dot"></span>
      <span id="hw-info">Apple Silicon (M4 Pro) MPS Acceleration</span>
    </div>
  </header>

  <main>
    <aside class="sidebar card">
      <h2>Architecture Specs</h2>
      <div class="spec-item"><span class="spec-label">Total Parameters</span><span class="spec-value">~1,070,352 (1.07M)</span></div>
      <div class="spec-item"><span class="spec-label">Hidden Dim (d_model)</span><span class="spec-value">144</span></div>
      <div class="spec-item"><span class="spec-label">Layers (Blocks)</span><span class="spec-value">4</span></div>
      <div class="spec-item"><span class="spec-label">Attention Heads</span><span class="spec-value">6 (24 dim/head)</span></div>
      <div class="spec-item"><span class="spec-label">Context Window</span><span class="spec-value">256 tokens</span></div>
      <div class="spec-item"><span class="spec-label">Tokenizer</span><span class="spec-value">Byte-Level (256 UTF-8)</span></div>

      <div class="control-group">
        <div class="control-label">
          <span>Temperature</span>
          <span id="temp-val" class="spec-value">0.8</span>
        </div>
        <input type="range" id="temp-slider" min="0.1" max="1.5" step="0.05" value="0.8">
      </div>

      <div class="control-group">
        <div class="control-label">
          <span>Top-K Sampling</span>
          <span id="topk-val" class="spec-value">40</span>
        </div>
        <input type="range" id="topk-slider" min="5" max="100" step="5" value="40">
      </div>

      <div class="control-group">
        <div class="control-label">
          <span>Max New Tokens</span>
          <span id="tokens-val" class="spec-value">250</span>
        </div>
        <input type="range" id="tokens-slider" min="50" max="500" step="25" value="250">
      </div>

      <div style="margin-top: 1.4rem;">
        <span class="control-label">Quick Prompts:</span>
        <div class="presets">
          <button class="preset-btn" onclick="setPrompt('ROMEO:')">ROMEO:</button>
          <button class="preset-btn" onclick="setPrompt('To be, or not to be')">To be, or not</button>
          <button class="preset-btn" onclick="setPrompt('CITIZEN:')">CITIZEN:</button>
          <button class="preset-btn" onclick="setPrompt('Once upon a time ')">Once upon a time</button>
          <button class="preset-btn" onclick="setPrompt('The king commanded ')">The king</button>
        </div>
      </div>
    </aside>

    <section class="playground card">
      <h2>Interactive Model Playground</h2>
      <div class="chat-output" id="output">Waiting for prompt... Click 'Generate' or select a preset prompt to test your 1M model.</div>
      <div class="input-row">
        <input type="text" id="prompt-input" class="prompt-input" placeholder="Type prompt here (e.g. ROMEO: or Once upon a time)..." value="ROMEO:">
        <button id="gen-btn" class="generate-btn" onclick="generate()">
          <span>⚡ Generate</span>
        </button>
      </div>
      <div class="status-bar">
        <span id="status-text">Status: Ready</span>
        <span id="speed-text">Speed: -- tok/s</span>
      </div>
    </section>
  </main>

  <script>
    const tempSlider = document.getElementById('temp-slider');
    const tempVal = document.getElementById('temp-val');
    const topkSlider = document.getElementById('topk-slider');
    const topkVal = document.getElementById('topk-val');
    const tokensSlider = document.getElementById('tokens-slider');
    const tokensVal = document.getElementById('tokens-val');
    const promptInput = document.getElementById('prompt-input');
    const genBtn = document.getElementById('gen-btn');
    const outputDiv = document.getElementById('output');
    const statusText = document.getElementById('status-text');
    const speedText = document.getElementById('speed-text');

    tempSlider.oninput = () => tempVal.textContent = tempSlider.value;
    topkSlider.oninput = () => topkVal.textContent = topkSlider.value;
    tokensSlider.oninput = () => tokensVal.textContent = tokensSlider.value;

    function setPrompt(p) {
      promptInput.value = p;
      promptInput.focus();
    }

    async function generate() {
      const prompt = promptInput.value.trim();
      if (!prompt) return;

      genBtn.disabled = true;
      statusText.textContent = "Generating with 1M model on M4 Pro...";
      outputDiv.textContent = prompt;

      const t0 = performance.now();
      try {
        const res = await fetch('/api/generate', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            prompt: prompt,
            temperature: parseFloat(tempSlider.value),
            top_k: parseInt(topkSlider.value),
            max_tokens: parseInt(tokensSlider.value)
          })
        });

        const data = await res.json();
        const dt = (performance.now() - t0) / 1000;
        if (data.error) {
          outputDiv.textContent += "\\n\\n[Notice: " + data.error + "]";
          statusText.textContent = "Model checkpoint not found yet. Train the model first!";
        } else {
          outputDiv.textContent = data.generated_text;
          const tokCount = data.new_tokens || 200;
          const tokRate = Math.round(tokCount / dt);
          speedText.textContent = `Speed: ~${tokRate} tok/s (${dt.toFixed(2)}s)`;
          statusText.textContent = "Generation complete!";
        }
      } catch (err) {
        outputDiv.textContent += "\\n\\n[Network error: " + err.message + "]";
        statusText.textContent = "Error during generation";
      } finally {
        genBtn.disabled = false;
      }
    }

    promptInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') generate();
    });
  </script>
</body>
</html>
"""


class StudioHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        elif self.path == "/api/status":
            try_load_model()
            data = {
                "model_loaded": model is not None,
                "device": str(device),
                "device_info": get_device_info(),
                "checkpoint_exists": os.path.exists(MODEL_PATH)
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode("utf-8"))
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        if self.path == "/api/generate":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            payload = json.loads(body)

            prompt = payload.get("prompt", "ROMEO:")
            temperature = float(payload.get("temperature", 0.8))
            top_k = int(payload.get("top_k", 40))
            max_tokens = int(payload.get("max_tokens", 200))

            try_load_model()
            if model is None:
                resp = {
                    "error": "No trained checkpoint found at 'model_1m.pt'. Run 'python train.py' first!",
                    "generated_text": prompt
                }
            else:
                encoded = tokenizer.encode(prompt)
                idx = torch.tensor([encoded], dtype=torch.long, device=device)
                with torch.no_grad():
                    gen_ids = model.generate(idx, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k)
                generated_text = tokenizer.decode(gen_ids[0])
                resp = {
                    "generated_text": generated_text,
                    "new_tokens": max_tokens
                }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))
        else:
            self.send_error(404, "Not Found")


def run_server(port: int = PORT):
    try_load_model()
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", port), StudioHandler) as httpd:
        print(f"\n🎉 NanoGPT-1M Studio Web UI running at http://localhost:{port}")
        print(f"Device: {get_device_info()} | Checkpoint: {MODEL_PATH}")
        print("Press Ctrl+C to stop.\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")


if __name__ == "__main__":
    port = PORT
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    run_server(port)
