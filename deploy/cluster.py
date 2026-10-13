"""Validate an operator-sized replica pool and render a secret-free gateway bundle.

Replica/GPU count is independent of the classroom's concurrent-user target.
"""

import ipaddress
import json
import re
from pathlib import Path
from urllib.parse import urlsplit

NAME = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9_-]{0,31}$')
DNS = re.compile(r'^(?=.{1,253}$)[a-zA-Z0-9](?:[a-zA-Z0-9.-]*[a-zA-Z0-9])?$')
GPU_DEVICE = re.compile(
    r'(?:0|[1-9][0-9]*|GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})',
    re.IGNORECASE,
)


def load_inventory(value):
    if not isinstance(value, dict) or set(value) != {'nodes'}:
        raise ValueError('Inventory must be an object containing only a nodes array')
    nodes = value['nodes']
    if not isinstance(nodes, list) or not nodes:
        raise ValueError('Inventory must contain at least one inference replica')
    result, names, addresses, devices = [], set(), set(), set()
    for node in nodes:
        if not isinstance(node, dict) or not {'name', 'url'} <= set(node) <= {
            'name',
            'url',
            'bind_host',
            'gpu_device',
        }:
            raise ValueError(
                'Each replica requires name and url; optional bind_host and gpu_device are supported'
            )
        name, url = node['name'], node['url']
        if not isinstance(name, str) or not NAME.fullmatch(name) or name.lower() in names:
            raise ValueError('Node names must be unique, simple identifiers of up to 32 characters')
        if not isinstance(url, str) or any(c.isspace() or ord(c) < 32 for c in url):
            raise ValueError('Node URLs cannot contain whitespace/control characters')
        parsed = urlsplit(url)
        host, port = parsed.hostname, parsed.port
        if (
            parsed.scheme != 'http'
            or not host
            or port is None
            or not 1 <= port <= 65535
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in ('', '/')
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                'Use http://HOST:PORT origins on the private/VPN network, without credentials or paths'
            )
        try:
            address = ipaddress.ip_address(host)
        except ValueError as exc:
            if not DNS.fullmatch(host) or not all(
                re.fullmatch(r'[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?', label)
                for label in host.split('.')
            ):
                raise ValueError('Invalid node hostname/address') from exc
            default_bind = '127.0.0.1'
        else:
            if address.is_unspecified or address.is_multicast:
                raise ValueError('Inventory requires a routable host address')
            host = str(address)
            default_bind = host
        bind = str(ipaddress.ip_address(node.get('bind_host', default_bind)))
        gpu_device = node.get('gpu_device', '0')
        if not isinstance(gpu_device, str) or not GPU_DEVICE.fullmatch(gpu_device):
            raise ValueError('gpu_device must be a decimal device index or a full GPU UUID string')
        if gpu_device.lower().startswith('gpu-'):
            gpu_device = 'GPU-' + gpu_device[4:].lower()
        target = f'[{host}]:{port}' if ':' in host else f'{host.lower()}:{port}'
        if target in addresses:
            raise ValueError('Replica endpoints must be unique')
        device = (host.lower(), gpu_device)
        if device in devices:
            raise ValueError('Replicas on the same host must use distinct GPU devices')
        names.add(name.lower())
        addresses.add(target)
        devices.add(device)
        result.append({'name': name, 'url': f'http://{target}', 'bind_host': bind, 'gpu_device': gpu_device})
    return result


def render_nginx(nodes):
    servers = '\n'.join(
        f'        server {urlsplit(node["url"]).netloc} max_fails=2 fail_timeout=10s;' for node in nodes
    )
    return (
        """worker_processes auto;
pid /tmp/nginx.pid;
error_log /dev/stderr warn;
events { worker_connections 1024; }
http {
    access_log off;
    client_body_temp_path /tmp/client_body;
    proxy_temp_path /tmp/proxy;
    upstream inference_fleet {
        zone inference_fleet 256k;
        least_conn;
"""
        + servers
        + """
        keepalive 64;
    }
    server {
        listen 8080;
        server_tokens off;
        client_max_body_size 1m;
        client_body_timeout 15s;
        send_timeout 120s;
        location = /health {
            proxy_pass http://fleet-health:8081/health;
            proxy_connect_timeout 2s;
            proxy_read_timeout 3s;
            access_log off;
        }
        location = /v1/chat/completions {
            limit_except POST { deny all; }
            proxy_pass http://inference_fleet;
            proxy_http_version 1.1;
            proxy_set_header Connection "";
            proxy_set_header Authorization $http_authorization;
            proxy_set_header Host $proxy_host;
            proxy_set_header X-Request-ID $request_id;
            proxy_buffering off;
            proxy_cache off;
            proxy_ignore_client_abort off;
            proxy_connect_timeout 3s;
            proxy_send_timeout 15s;
            proxy_read_timeout 120s;
            proxy_next_upstream error timeout http_502 http_503 http_504;
            proxy_next_upstream_tries 3;
            proxy_next_upstream_timeout 5s;
            add_header X-Accel-Buffering no always;
        }
        location / { return 404; }
    }
}
"""
    )


def generate_bundle(inventory: Path, output: Path):
    nodes = load_inventory(json.loads(inventory.read_text(encoding='utf-8')))
    # mkdir is exclusive: never replace an operator's existing configuration.
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    (output / 'nginx.conf').write_text(render_nginx(nodes), encoding='utf-8')
    (output / 'nodes.json').write_text(json.dumps({'nodes': nodes}, indent=2) + '\n', encoding='utf-8')
    for node in nodes:
        port = urlsplit(node['url']).port
        bind_host = node['bind_host']
        (output / f'{node["name"]}.env').write_text(
            '# Non-secret node settings; supply VLLM_API_KEY separately on the node.\n'
            f'# Gateway reaches this node at {node["url"]}.\n'
            '# DNS inventories default to loopback binding: set the private interface IP before rollout.\n'
            f'COMPOSE_PROJECT_NAME=teacher-{node["name"].lower()}\n'
            f'VLLM_BIND_HOST={bind_host}\nVLLM_PORT={port}\n'
            f'VLLM_GPU_DEVICE={node["gpu_device"]}\n'
            'VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct\nVLLM_MAX_MODEL_LEN=16384\n'
            'VLLM_MAX_NUM_SEQS=4\nVLLM_GPU_MEMORY_UTILIZATION=0.88\n',
            encoding='utf-8',
        )
    return len(nodes)
