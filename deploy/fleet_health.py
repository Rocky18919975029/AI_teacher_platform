"""Probe every configured inference replica; serve bounded, stale-safe readiness."""

import json
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from deploy.cluster import load_inventory


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def probe_node(node, key, model, timeout=2):
    # Ignore proxy environment variables and never forward credentials on a redirect.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    headers = {'Authorization': f'Bearer {key}'}
    try:
        with opener.open(
            urllib.request.Request(node['url'] + '/health', headers=headers), timeout=timeout
        ) as reply:
            if reply.status != 200:
                return False
        with opener.open(
            urllib.request.Request(node['url'] + '/v1/models', headers=headers), timeout=timeout
        ) as reply:
            data = json.loads(reply.read(65537))
            return reply.status == 200 and any(row.get('id') == model for row in data.get('data', []))
    except (OSError, ValueError, TypeError, AttributeError):
        return False


class FleetState:
    def __init__(self, total, minimum, max_age=30):
        if not 1 <= minimum <= total:
            raise ValueError('Minimum ready nodes must be between one and fleet size')
        self.total, self.minimum, self.max_age = total, minimum, max_age
        self.checked_at = 0.0
        self.healthy = 0
        self.lock = threading.Lock()

    def update(self, statuses):
        with self.lock:
            self.healthy = sum(bool(value) for value in statuses)
            self.checked_at = time.monotonic()

    def snapshot(self):
        with self.lock:
            fresh = self.checked_at > 0 and time.monotonic() - self.checked_at <= self.max_age
            return {
                'ready': fresh and self.healthy >= self.minimum,
                'healthy_nodes': self.healthy,
                'total_nodes': self.total,
                'required_nodes': self.minimum,
                'fresh': fresh,
            }


def main():
    key = os.environ.get('VLLM_API_KEY', '')
    if len(key) < 32 or 'CHANGE_ME' in key or key == 'EMPTY':
        raise SystemExit('A non-placeholder VLLM_API_KEY of at least 32 characters is required')
    nodes = load_inventory(json.loads(Path(os.environ.get('NODES_FILE', '/config/nodes.json')).read_text()))
    model = os.environ.get('VLLM_MODEL', 'Qwen/Qwen2.5-7B-Instruct')
    state = FleetState(len(nodes), len(nodes))

    def refresh():
        with ThreadPoolExecutor(max_workers=20, thread_name_prefix='fleet-probe') as executor:
            while True:
                try:
                    state.update(list(executor.map(lambda node: probe_node(node, key, model), nodes)))
                except Exception:
                    state.update([False] * len(nodes))
                time.sleep(5)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != '/health':
                self.send_error(404)
                return
            data = state.snapshot()
            body = json.dumps(data).encode()
            self.send_response(200 if data['ready'] else 503)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    threading.Thread(target=refresh, daemon=True, name='fleet-refresh').start()
    HTTPServer(('0.0.0.0', 8081), Handler).serve_forever()


if __name__ == '__main__':
    main()
