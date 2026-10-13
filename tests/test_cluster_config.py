import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from deploy.cluster import generate_bundle, load_inventory, render_nginx
from deploy.fleet_health import FleetState, probe_node


def inventory(count=40):
    return {'nodes': [{'name': f'gpu{i:02}', 'url': f'http://10.23.0.{i}:8000'} for i in range(1, count + 1)]}


@pytest.mark.parametrize('count', [1, 2, 8, 40, 64])
def test_render_any_nonempty_replica_count_with_stream_safe_failover(count):
    nodes = load_inventory(inventory(count))
    config = render_nginx(nodes)
    assert len(nodes) == count
    assert config.count('max_fails=2 fail_timeout=10s;') == count
    assert 'least_conn;' in config and 'zone inference_fleet' in config
    assert 'proxy_buffering off;' in config and 'proxy_ignore_client_abort off;' in config
    assert 'proxy_next_upstream error timeout http_502 http_503 http_504;' in config
    assert 'non_idempotent' not in config
    assert 'proxy_pass http://fleet-health:8081/health;' in config
    assert 'Authorization $http_authorization' in config


@pytest.mark.parametrize(
    'bad_url',
    [
        'http://user:secret@host:8000',
        'http://host:8000/path',
        'http://host:8000?key=secret',
        'http://host:8000#fragment',
        'http://host:8000;server evil',
        'http://host:\n8000',
        'https://host:8000',
        'http://host',
        'http://host:65536',
        'http://0.0.0.0:8000',
        'http://224.0.0.1:8000',
        'http://bad..host:8000',
    ],
)
def test_reject_unsafe_node_urls(bad_url):
    data = inventory()
    data['nodes'][0]['url'] = bad_url
    with pytest.raises(ValueError):
        load_inventory(data)


def test_count_duplicates_and_dns_binding():
    data = inventory()
    with pytest.raises(ValueError):
        load_inventory({'nodes': []})
    data['nodes'][1]['url'] = data['nodes'][0]['url']
    with pytest.raises(ValueError):
        load_inventory(data)
    data = inventory()
    data['nodes'][1]['name'] = data['nodes'][0]['name']
    with pytest.raises(ValueError):
        load_inventory(data)
    data = inventory()
    data['nodes'][0] = {'name': 'dns-node', 'url': 'http://gpu01.example:8000'}
    assert load_inventory(data)[0]['bind_host'] == '127.0.0.1'
    data['nodes'][0]['bind_host'] = '10.23.0.1'
    assert load_inventory(data)[0]['bind_host'] == '10.23.0.1'


def test_generator_never_overwrites_and_contains_no_credentials(tmp_path):
    source, output = tmp_path / 'nodes.json', tmp_path / 'bundle'
    source.write_text(json.dumps(inventory()))
    assert generate_bundle(source, output) == 40
    assert len(list(output.glob('gpu*.env'))) == 40
    original = (output / 'nginx.conf').read_text()
    with pytest.raises(FileExistsError):
        generate_bundle(source, output)
    assert (output / 'nginx.conf').read_text() == original
    assert 'VLLM_API_KEY=' not in ''.join(p.read_text() for p in output.glob('*'))
    assert 'VLLM_BIND_HOST=10.23.0.1' in (output / 'gpu01.env').read_text()
    assert 'VLLM_GPU_DEVICE=0\n' in (output / 'gpu01.env').read_text()
    assert 'COMPOSE_PROJECT_NAME=teacher-gpu01\n' in (output / 'gpu01.env').read_text()


def test_two_gpus_on_one_host_get_distinct_devices_ports_and_compose_projects(tmp_path):
    data = {
        'nodes': [
            {'name': 'GPU-A', 'url': 'http://10.23.0.1:8000', 'gpu_device': '0'},
            {'name': 'GPU-B', 'url': 'http://10.23.0.1:8001', 'gpu_device': '1'},
        ]
    }
    nodes = load_inventory(data)
    assert [node['gpu_device'] for node in nodes] == ['0', '1']
    assert len({node['url'] for node in nodes}) == 2
    source, output = tmp_path / 'nodes.json', tmp_path / 'bundle'
    source.write_text(json.dumps(data))
    assert generate_bundle(source, output) == 2
    first = (output / 'GPU-A.env').read_text()
    second = (output / 'GPU-B.env').read_text()
    assert 'VLLM_GPU_DEVICE=0\n' in first and 'VLLM_GPU_DEVICE=1\n' in second
    assert 'VLLM_PORT=8000\n' in first and 'VLLM_PORT=8001\n' in second
    assert 'COMPOSE_PROJECT_NAME=teacher-gpu-a\n' in first
    assert 'COMPOSE_PROJECT_NAME=teacher-gpu-b\n' in second
    assert load_inventory(json.loads((output / 'nodes.json').read_text())) == nodes


def test_omitted_gpu_device_defaults_to_zero_and_is_valid_on_distinct_hosts():
    nodes = load_inventory(inventory(2))
    assert [node['gpu_device'] for node in nodes] == ['0', '0']


@pytest.mark.parametrize('device', ['0', '1', '12'])
def test_canonical_decimal_gpu_indices_are_accepted(device):
    data = inventory(1)
    data['nodes'][0]['gpu_device'] = device
    assert load_inventory(data)[0]['gpu_device'] == device


@pytest.mark.parametrize(
    'device',
    [
        '',
        '00',
        '01',
        '+1',
        '-1',
        '0.0',
        '0,1',
        'all',
        'none',
        'GPU-short',
        'MIG-12345678-1234-1234-1234-123456789abc',
        '0\nVLLM_API_KEY=injected',
        '$(command)',
        '${INJECT}',
        ' 0',
        '0 ',
        0,
        1,
        None,
        True,
        [],
        {},
    ],
)
def test_reject_invalid_or_injected_gpu_selectors(device):
    data = inventory(1)
    data['nodes'][0]['gpu_device'] = device
    with pytest.raises(ValueError):
        load_inventory(data)


@pytest.mark.parametrize(
    'first_url,second_url',
    [
        ('http://gpu.example:8000', 'http://GPU.EXAMPLE:8001'),
        ('http://10.23.0.1:8000', 'http://10.23.0.1:8001'),
        ('http://[fd00::1]:8000', 'http://[fd00:0:0:0:0:0:0:1]:8001'),
    ],
)
def test_same_normalized_host_and_gpu_rejected_even_with_different_ports(first_url, second_url):
    with pytest.raises(ValueError):
        load_inventory(
            {
                'nodes': [
                    {'name': 'first', 'url': first_url},
                    {'name': 'second', 'url': second_url, 'gpu_device': '0'},
                ]
            }
        )


def test_distinct_devices_still_require_distinct_endpoints():
    with pytest.raises(ValueError):
        load_inventory(
            {
                'nodes': [
                    {'name': 'first', 'url': 'http://10.23.0.1:8000', 'gpu_device': '0'},
                    {'name': 'second', 'url': 'http://10.23.0.1:8000', 'gpu_device': '1'},
                ]
            }
        )


def test_full_gpu_uuid_supported_and_written_to_environment(tmp_path):
    device = 'GPU-12345678-abcd-1234-abcd-123456789abc'
    data = inventory(1)
    data['nodes'][0]['gpu_device'] = device
    nodes = load_inventory(data)
    assert nodes[0]['gpu_device'].lower() == device.lower()
    source, output = tmp_path / 'nodes.json', tmp_path / 'bundle'
    source.write_text(json.dumps(data))
    assert generate_bundle(source, output) == 1
    assert f'VLLM_GPU_DEVICE={nodes[0]["gpu_device"]}\n' in (output / 'gpu01.env').read_text()


def test_uuid_case_does_not_bypass_duplicate_gpu_detection():
    uuid = '12345678-abcd-1234-abcd-123456789abc'
    with pytest.raises(ValueError):
        load_inventory(
            {
                'nodes': [
                    {'name': 'first', 'url': 'http://10.23.0.1:8000', 'gpu_device': f'GPU-{uuid}'},
                    {'name': 'second', 'url': 'http://10.23.0.1:8001', 'gpu_device': f'GPU-{uuid.upper()}'},
                ]
            }
        )


def test_ipv6_uses_brackets_only_for_upstream_not_compose_host_ip(tmp_path):
    data = inventory()
    data['nodes'][0] = {'name': 'ipv6', 'url': 'http://[fd00::1]:8000'}
    nodes = load_inventory(data)
    assert nodes[0]['bind_host'] == 'fd00::1'
    assert 'server [fd00::1]:8000 ' in render_nginx(nodes)
    source, output = tmp_path / 'nodes.json', tmp_path / 'bundle'
    source.write_text(json.dumps(data))
    generate_bundle(source, output)
    assert 'VLLM_BIND_HOST=fd00::1' in (output / 'ipv6.env').read_text()


def test_fleet_readiness_failclosed_and_stale():
    state = FleetState(40, 40, max_age=10)
    assert not state.snapshot()['ready']
    state.update([True] * 40)
    assert state.snapshot()['ready']
    state.update([True] * 39 + [False])
    assert not state.snapshot()['ready']
    state.update([True] * 40)
    state.checked_at = time.monotonic() - 11
    assert not state.snapshot()['ready']
    assert not state.snapshot()['fresh']
    with pytest.raises(ValueError):
        FleetState(40, 41)


def test_probe_requires_authenticated_matching_model_without_following_redirects():
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append(self.path)
            if self.headers.get('Authorization') != 'Bearer test-inference-secret':
                self.send_error(401)
                return
            if self.path == '/health':
                self.send_response(200)
                self.end_headers()
            elif self.path == '/v1/models':
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"data":[{"id":"expected-model"}]}')
            else:
                self.send_response(302)
                self.send_header('Location', '/health')
                self.end_headers()

        def log_message(self, *args):
            pass

    with HTTPServer(('127.0.0.1', 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        node = {'url': f'http://127.0.0.1:{server.server_port}'}
        try:
            assert probe_node(node, 'test-inference-secret', 'expected-model')
            assert not probe_node(node, 'wrong-key', 'expected-model')
            assert not probe_node(node, 'test-inference-secret', 'wrong-model')
            before = len(seen)
            assert not probe_node(
                {'url': node['url'] + '/redirect'}, 'test-inference-secret', 'expected-model'
            )
            assert len(seen) == before + 1
        finally:
            server.shutdown()
            thread.join(timeout=2)
