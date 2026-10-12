"""Upload an instructor-owned file and wait for its durable ingestion job."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('file', type=Path)
    parser.add_argument('--course', required=True)
    parser.add_argument('--url', default='http://127.0.0.1:9000')
    parser.add_argument('--wait-seconds', type=int, default=600)
    args = parser.parse_args()
    key = os.environ.get('TEACHER_API_KEY', '')
    if not key or not args.file.is_file() or args.wait_seconds < 1:
        parser.error('Set TEACHER_API_KEY, provide an existing file and a positive wait timeout')
    try:
        with httpx.Client(
            base_url=args.url.rstrip('/'),
            headers={'Authorization': f'Bearer {key}'},
            timeout=60,
            follow_redirects=False,
        ) as client:
            with args.file.open('rb') as source:
                response = client.post(
                    '/api/documents',
                    params={'course_id': args.course},
                    files={'file': (args.file.name, source)},
                )
            response.raise_for_status()
            job = response.json()
            deadline = time.monotonic() + args.wait_seconds
            while job['status'] in {'pending', 'running'}:
                if time.monotonic() >= deadline:
                    print(json.dumps({'job_id': job['id'], 'status': 'poll_timeout'}))
                    return 2
                time.sleep(min(2, max(0, deadline - time.monotonic())))
                response = client.get(f'/api/jobs/{job["id"]}')
                response.raise_for_status()
                job = response.json()
            print(json.dumps(job, ensure_ascii=False, indent=2))
            return 0 if job['status'] == 'completed' else 1
    except (httpx.HTTPError, ValueError, KeyError, OSError) as exc:
        # Avoid printing credentials, document contents or proxy response bodies.
        print(f'Ingestion failed ({type(exc).__name__}); inspect the job/API logs.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
