"""Abre la página web en tu navegador:  python serve.py  [--port 8000]

Sirve la carpeta web/ en http://localhost:8000. Hace falta un servidor (y no abrir el
.html con doble clic) para que la página pueda cargar la red de web/model/.
"""
import argparse
import functools
import http.server
import os
import socketserver
import threading
import webbrowser

WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      ".js": "application/javascript", ".json": "application/json", ".bin": "application/octet-stream"}

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer(("127.0.0.1", args.port), functools.partial(Handler, directory=WEB)) as httpd:
        url = f"http://localhost:{args.port}/"
        has_model = os.path.exists(os.path.join(WEB, "model", "model.json"))
        print(f"Análisis Rápido FAMACHA → {url}")
        print("Red FAMACHA-Net: " + ("se carga desde web/model/" if has_model else "no hay modelo en web/model/ (se usa el motor colorimétrico)"))
        print("Ctrl+C para detener.")
        if not args.no_browser:
            threading.Timer(0.6, lambda: webbrowser.open(url)).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
