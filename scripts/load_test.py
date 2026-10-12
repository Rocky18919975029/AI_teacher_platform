#!/usr/bin/env python3
"""Synchronized real-service load acceptance; never substitute mock results for GPU evidence.

Use a mode-0600 JSON array of {"key": "...", "course_id": "physics"} entries
for distinct students. TEACHER_API_KEY is a shared-identity fallback, subject to
one principal's rate limit. Prompts, credentials, answer text and session IDs
are not written to the report. Real chat requests persist server-side sessions.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import stat
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import httpx


@dataclass(frozen=True)
class Student:
    key: str
    course_id: str


@dataclass
class Sample:
    kind: str
    success: bool
    total_seconds: float
    ttft_seconds: float | None = None
    error: str | None = None
    source_count: int = 0


class InvalidStream(ValueError):
    pass


def load_students(path: Path | None, count: int, course: str):
    if path:
        # Do not accidentally encourage readable credential inventories.
        if os.name == 'posix' and stat.S_IMODE(path.stat().st_mode) & 0o077:
            raise ValueError('Users file must be private (chmod 600).')
        raw = json.loads(path.read_text())
        if not isinstance(raw, list) or len(raw) < count:
            raise ValueError('Users file must contain at least --users entries.')
        if any(not isinstance(row, dict) for row in raw[:count]):
            raise ValueError('Each user must be an object with key and optional course_id.')
        students = [Student(row.get('key', ''), row.get('course_id', course)) for row in raw[:count]]
        if len({s.key for s in students if isinstance(s.key, str)}) != count:
            raise ValueError('Users file must use a distinct key for each simulated student.')
        identity = 'distinct_keys_verify_distinct_server_subjects'
    else:
        students = [Student(os.environ.get('TEACHER_API_KEY', ''), course)] * count
        identity = 'shared_key_not_distinct_student_identities'
    if any(not isinstance(s.key, str) or len(s.key) < 32 or '\n' in s.key or '\r' in s.key for s in students):
        raise ValueError('Supply valid keys of at least 32 characters via --users-file or TEACHER_API_KEY.')
    if any(
        not isinstance(s.course_id, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', s.course_id)
        for s in students
    ):
        raise ValueError('Invalid course_id in workload.')
    return students, identity


async def sse_events(response):
    event, data, size = 'message', [], 0
    async for line in response.aiter_lines():
        if not line:
            if data:
                try:
                    payload = json.loads('\n'.join(data))
                except json.JSONDecodeError:
                    raise InvalidStream('invalid_event_json') from None
                if not isinstance(payload, dict):
                    raise InvalidStream('invalid_event_payload')
                yield event, payload
            event, data, size = 'message', [], 0
        elif not line.startswith(':'):
            field, _, value = line.partition(':')
            value = value.removeprefix(' ')
            if field == 'event':
                event = value
            elif field == 'data':
                size += len(value)
                if size > 262144:
                    raise InvalidStream('event_too_large')
                data.append(value)
    if data:
        raise InvalidStream('truncated_event')


def source_count(value):
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise InvalidStream('invalid_sources')
    return len(value)


async def perform(client, student, prompt, kind, *, timeout, require_sources, session=None):
    start = time.perf_counter()
    sample = Sample(kind, False, 0)
    sid = None
    headers = {'Authorization': f'Bearer {student.key}'}
    try:
        async with asyncio.timeout(timeout):
            if kind == 'search':
                response = await client.post(
                    '/api/search', headers=headers, json={'course_id': student.course_id, 'query': prompt}
                )
                if response.status_code != 200:
                    raise InvalidStream(f'http_{response.status_code}')
                payload = response.json()
                if not isinstance(payload, dict):
                    raise InvalidStream('invalid_search_payload')
                sample.source_count = source_count(payload.get('sources'))
                if require_sources and not sample.source_count:
                    raise InvalidStream('no_sources')
            else:
                payload = {'course_id': student.course_id, 'prompt': prompt, 'rag': True, 'with_audio': False}
                if session:
                    payload['session_id'] = session
                async with client.stream(
                    'POST', '/api/chat/stream', headers=headers, json=payload
                ) as response:
                    if response.status_code != 200:
                        raise InvalidStream(f'http_{response.status_code}')
                    if not response.headers.get('content-type', '').startswith('text/event-stream'):
                        raise InvalidStream('not_event_stream')
                    saw_done = False
                    answer = ''
                    async for event, data in sse_events(response):
                        if saw_done:
                            raise InvalidStream('event_after_done')
                        if event == 'error':
                            raise InvalidStream('sse_error')
                        if event == 'token':
                            token = data.get('text')
                            if not isinstance(token, str):
                                raise InvalidStream('invalid_token')
                            if token and sample.ttft_seconds is None:
                                sample.ttft_seconds = time.perf_counter() - start
                            answer += token
                            if len(answer) > 65536:
                                raise InvalidStream('answer_too_large')
                        elif event == 'sources':
                            sample.source_count = source_count(data.get('sources'))
                        elif event == 'done':
                            if not answer.strip() or data.get('text') != answer:
                                raise InvalidStream('missing_or_inconsistent_answer')
                            sample.source_count = source_count(data.get('sources'))
                            # Empty-grounding fallback is not a successful LLM request,
                            # even though the API legitimately ends it with HTTP 200/done.
                            if not sample.source_count:
                                raise InvalidStream('ungrounded_fallback')
                            sid = data.get('session_id')
                            if not isinstance(sid, str) or not sid or len(sid) > 36:
                                raise InvalidStream('invalid_session')
                            saw_done = True
                    if not saw_done:
                        raise InvalidStream('missing_done')
            sample.success = True
    except InvalidStream as exc:
        sample.error = str(exc)  # Only locally defined categories, never response bodies.
    except (TimeoutError, httpx.TimeoutException):
        sample.error = 'timeout'
    except httpx.HTTPError:
        sample.error = 'transport_error'
    except (ValueError, TypeError):
        sample.error = 'invalid_response'
    sample.total_seconds = time.perf_counter() - start
    return sample, sid


def percentile(values, percent):
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(len(ordered) * percent / 100) - 1)], 6)


def summarize(samples, elapsed):
    successful = [s for s in samples if s.success]
    ttfts = [s.ttft_seconds for s in successful if s.ttft_seconds is not None]
    totals = [s.total_seconds for s in successful]
    return {
        'requests': len(samples),
        'successes': len(successful),
        'failures': len(samples) - len(successful),
        'errors': dict(Counter(s.error for s in samples if not s.success)),
        'success_rate': len(successful) / len(samples) if samples else None,
        'elapsed_seconds': round(elapsed, 6),
        'successful_requests_per_second': round(len(successful) / elapsed, 6) if elapsed else None,
        'ttft_seconds': {'p50': percentile(ttfts, 50), 'p95': percentile(ttfts, 95), 'samples': len(ttfts)},
        'total_seconds': {
            'p50': percentile(totals, 50),
            'p95': percentile(totals, 95),
            'samples': len(totals),
        },
        'sources_returned': sum(s.source_count for s in successful),
    }


async def run_load(
    client,
    students,
    prompts,
    *,
    mode='chat',
    rounds=3,
    warmup_rounds=1,
    timeout=180,
    require_sources=False,
    max_p95_ttft=None,
    max_p95_total=None,
):
    sessions = [None] * len(students)

    async def phase(count, offset):
        samples = []
        started = time.perf_counter()
        for round_index in range(count):
            gate = asyncio.Event()

            async def one(index):
                kind = 'search' if mode == 'search' or (mode == 'mixed' and index % 2) else 'chat'
                prompt = prompts[((offset + round_index) * len(students) + index) % len(prompts)]
                await gate.wait()
                sample, sid = await perform(
                    client,
                    students[index],
                    prompt,
                    kind,
                    timeout=timeout,
                    require_sources=require_sources,
                    session=sessions[index],
                )
                if sample.success and sid:
                    sessions[index] = sid
                return sample

            tasks = [asyncio.create_task(one(index)) for index in range(len(students))]
            await asyncio.sleep(0)  # All tasks register at the start barrier before release.
            gate.set()
            samples.extend(await asyncio.gather(*tasks))
        return samples, time.perf_counter() - started

    warmup, warmup_elapsed = await phase(warmup_rounds, 0)
    measured, elapsed = await phase(rounds, warmup_rounds)
    summary = summarize(measured, elapsed)
    by_kind = {
        kind: summarize([s for s in measured if s.kind == kind], elapsed)
        for kind in sorted({s.kind for s in measured})
    }
    violations = []
    if any(not sample.success for sample in warmup):
        violations.append('warmup_failed')
    if summary['failures']:
        violations.append('success_rate_below_100_percent')
    for kind, stats in by_kind.items():
        if kind == 'chat' and max_p95_ttft is not None:
            value = stats['ttft_seconds']['p95']
            if value is None or value > max_p95_ttft:
                violations.append(f'{kind}_p95_ttft_exceeded_or_unavailable')
        if max_p95_total is not None:
            value = stats['total_seconds']['p95']
            if value is None or value > max_p95_total:
                violations.append(f'{kind}_p95_total_exceeded_or_unavailable')
    return {
        'schema_version': 1,
        'passed': not violations,
        'violations': violations,
        'workload': {
            'users': len(students),
            'mode': mode,
            'rounds': rounds,
            'warmup_rounds': warmup_rounds,
            'require_sources': require_sources,
            'audio': False,
            'synchronized_rounds': True,
            'request_timeout_seconds': timeout,
        },
        'thresholds': {
            'success_rate': 1.0,
            'max_p95_ttft_seconds': max_p95_ttft,
            'max_p95_total_seconds': max_p95_total,
        },
        'warmup': summarize(warmup, warmup_elapsed),
        'measured': summary,
        'by_kind': by_kind,
        'measurement_notes': [
            'TTFT includes retrieval and is measured at the client from request start to first text token.',
            'Latency percentiles include successful requests only; every failure fails acceptance.',
            'No latency SLO is asserted unless thresholds are supplied; audio is excluded.',
            'Chat always requires grounding; --require-sources additionally rejects empty search results.',
            'This workload creates persistent chat sessions and uses existing course documents.',
            'The CLI connects directly and ignores environment proxy configuration.',
        ],
    }


def positive_float(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError('Must be finite and positive.')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--users-file', type=Path)
    parser.add_argument('--users', type=int, default=40)
    parser.add_argument('--course-id', default='default')
    parser.add_argument('--mode', choices=['chat', 'search', 'mixed'], default='chat')
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--warmup-rounds', type=int, default=1)
    parser.add_argument('--prompt', default='Explain a key concept from the course and cite its source.')
    parser.add_argument('--prompts-file', type=Path, help='JSON array of representative course questions.')
    parser.add_argument('--timeout', type=positive_float, default=180)
    parser.add_argument(
        '--require-sources',
        action='store_true',
        help='Reject empty search results; chat always requires nonempty grounding.',
    )
    parser.add_argument('--max-p95-ttft', type=positive_float)
    parser.add_argument('--max-p95-total', type=positive_float)
    parser.add_argument(
        '--output', type=Path, help='Write a new JSON report; existing files are not overwritten.'
    )
    args = parser.parse_args(argv)
    url = urlsplit(args.base_url)
    if (
        url.scheme not in {'http', 'https'}
        or not url.hostname
        or url.username
        or url.password
        or url.query
        or url.fragment
    ):
        parser.error('--base-url must be an HTTP(S) URL without credentials, query or fragment.')
    if not 1 <= args.users <= 256 or not 1 <= args.rounds <= 1000 or not 0 <= args.warmup_rounds <= 100:
        parser.error('Require users 1..256, rounds 1..1000 and warmup-rounds 0..100.')
    if args.mode == 'search' and args.max_p95_ttft is not None:
        parser.error('TTFT is only defined for chat; use --max-p95-total for search.')
    if args.output and args.output.exists():
        parser.error('--output already exists; choose a new report path.')
    try:
        students, identity = load_students(args.users_file, args.users, args.course_id)
        prompts = json.loads(args.prompts_file.read_text()) if args.prompts_file else [args.prompt]
        if (
            not isinstance(prompts, list)
            or not prompts
            or any(not isinstance(p, str) or not p.strip() or len(p) > 8000 for p in prompts)
        ):
            raise ValueError('Prompts must be a nonempty JSON array of strings (1..8000 characters).')
    except (OSError, ValueError) as exc:
        # Do not print JSON source or credentials from invalid input.
        parser.error(str(exc) if not isinstance(exc, json.JSONDecodeError) else 'Invalid JSON input.')
    if not args.users_file:
        print(
            'Warning: TEACHER_API_KEY simulates a shared principal, not 40 distinct students; '
            'its per-subject rate limit applies.',
            file=sys.stderr,
        )

    async def execute():
        async with httpx.AsyncClient(
            base_url=args.base_url.rstrip('/'),
            timeout=args.timeout,
            trust_env=False,
            limits=httpx.Limits(max_connections=args.users + 8, max_keepalive_connections=args.users + 8),
        ) as client:
            return await run_load(
                client,
                students,
                prompts,
                mode=args.mode,
                rounds=args.rounds,
                warmup_rounds=args.warmup_rounds,
                timeout=args.timeout,
                require_sources=args.require_sources,
                max_p95_ttft=args.max_p95_ttft,
                max_p95_total=args.max_p95_total,
            )

    try:
        report = asyncio.run(execute())
        report['workload']['identity_mode'] = identity
        rendered = json.dumps(report, indent=2) + '\n'
        if args.output:
            with args.output.open('x') as output:
                output.write(rendered)
        else:
            print(rendered, end='')
        return 0 if report['passed'] else 1
    except KeyboardInterrupt:
        print('Load test interrupted; no acceptance result.', file=sys.stderr)
        return 130
    except OSError:
        print('Could not write the report; choose a writable new output path.', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
