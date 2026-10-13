"""Serve only this experiment's local artifacts on the loopback interface."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json,os

root=Path(__file__).resolve().parent
server=ThreadingHTTPServer(('127.0.0.1',0),partial(SimpleHTTPRequestHandler,directory=str(root)))
info=dict(pid=os.getpid(),url=f'http://127.0.0.1:{server.server_port}/index.html')
(root/'viewer_connection.json').write_text(json.dumps(info)+'\n')
print(json.dumps(info),flush=True)
server.serve_forever()
