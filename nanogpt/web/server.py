"""Minimal dependency-free HTTP server: one HTML page + JSON POST endpoints."""

import http.server
import json
import socketserver
from pathlib import Path
from typing import Callable, Dict

_TEMPLATES = Path(__file__).parent / "templates"


def load_template(name: str) -> str:
    return (_TEMPLATES / name).read_text(encoding="utf-8")


def serve(*, port: int, html: str, endpoints: Dict[str, Callable[[dict], dict]], host: str = "127.0.0.1"):
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_):  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
            else:
                self.send_error(404)

        def do_POST(self):
            fn = endpoints.get(self.path)
            if fn is None:
                self.send_error(404)
                return
            n = int(self.headers.get("Content-Length", 0))
            try:
                payload = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
            except Exception:
                self._send(400, b'{"error": "bad request"}', "application/json")
                return
            try:
                resp = fn(payload)
            except Exception as e:  # surface errors to the page instead of 500
                resp = {"error": str(e)}
            self._send(200, json.dumps(resp).encode("utf-8"), "application/json")

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer((host, port), Handler) as httpd:
        print(f"Serving on http://localhost:{port}  (Ctrl+C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")
