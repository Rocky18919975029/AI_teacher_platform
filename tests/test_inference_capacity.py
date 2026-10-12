"""Real TCP connection admission, not a GPU throughput benchmark."""

import asyncio
from contextlib import suppress

from teacher.config import Settings
from teacher.inference import Inference


def settings(tmp_path, **overrides):
    return Settings(
        _env_file=None,
        data_dir=tmp_path,
        api_keys=[{'key': 'capacity-test-key-' * 3, 'subject': 'test', 'courses': ['x']}],
        **overrides,
    )


async def test_40_real_upstream_streams_open_before_any_completes(tmp_path):
    received = 0
    all_received = asyncio.Event()
    release = asyncio.Event()
    handlers = set()

    async def upstream(reader, writer):
        nonlocal received
        task = asyncio.current_task()
        handlers.add(task)
        try:
            headers = await reader.readuntil(b'\r\n\r\n')
            length = next(
                int(line.split(b':', 1)[1])
                for line in headers.split(b'\r\n')
                if line.lower().startswith(b'content-length:')
            )
            await reader.readexactly(length)
            received += 1
            if received == 40:
                all_received.set()
            writer.write(b'HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n')
            writer.write(b'data: {"choices":[{"delta":{"content":"First"}}]}\n\n')
            await writer.drain()
            await release.wait()
            writer.write(b'data: [DONE]\n\n')
            await writer.drain()
        finally:
            writer.close()
            with suppress(ConnectionError):
                await writer.wait_closed()
            handlers.discard(task)

    server = await asyncio.start_server(upstream, '127.0.0.1', 0)
    port = server.sockets[0].getsockname()[1]
    inference = Inference(settings(tmp_path, vllm_url=f'http://127.0.0.1:{port}/v1/chat/completions'))

    async def consume():
        return [token async for token in inference.stream([{'role': 'user', 'content': 'test'}])]

    requests = [asyncio.create_task(consume()) for _ in range(40)]
    try:
        await asyncio.wait_for(all_received.wait(), timeout=5)
        assert received == 40
        assert all(not request.done() for request in requests)
        release.set()
        assert await asyncio.wait_for(asyncio.gather(*requests), timeout=5) == [['First']] * 40
    finally:
        release.set()
        for request in requests:
            request.cancel()
        await asyncio.gather(*requests, return_exceptions=True)
        await inference.close()
        server.close()
        await server.wait_closed()
        await asyncio.gather(*handlers, return_exceptions=True)


async def test_audio_burst_is_bounded_and_overload_does_not_leak_slots(tmp_path, monkeypatch):
    inference = Inference(settings(tmp_path, tts_enabled=True, max_concurrent_tts=2, tts_queue_timeout=0.05))
    active = 0
    peak = 0
    ready = asyncio.Event()
    release = asyncio.Event()

    async def synthesize(text, directory):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        if active == 2:
            ready.set()
        try:
            await release.wait()
            return 'audio-id'
        finally:
            active -= 1

    monkeypatch.setattr(inference, '_synthesize', synthesize)
    tasks = [asyncio.create_task(inference.synthesize('answer', tmp_path)) for _ in range(40)]
    try:
        await asyncio.wait_for(ready.wait(), timeout=1)
        await asyncio.sleep(0.1)
        release.set()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        assert peak == 2
        assert results.count('audio-id') == 2
        assert sum(isinstance(result, TimeoutError) for result in results) == 38
        assert await inference.synthesize('next', tmp_path) == 'audio-id'
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        await inference.close()
