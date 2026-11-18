"""Serves the evaluator UI (runs on Laptop 2).

    python ui/serve.py            # http://localhost:8501
    python ui/serve.py --port 9000
Open the page, type Laptop 1's API address (e.g. http://192.168.1.20:8000) and press "Load trips".
"""
import argparse
import functools
import http.server
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8501)
args = parser.parse_args()
handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(Path(__file__).parent))
with http.server.ThreadingHTTPServer(("0.0.0.0", args.port), handler) as srv:
    print(f"PathSnap UI on http://localhost:{args.port}  (other devices: http://<this-laptop-ip>:{args.port})")
    srv.serve_forever()
