"""Evaluate live retrieval using a labeled JSONL dataset; never invent quality scores.

Each row: {"course_id": "...", "query": "...", "relevant_document_ids": ["uuid"]}.
Scores use the first k returned chunks and their immutable document IDs.
"""

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import httpx


def score_ranking(document_ids: list[str], relevant: set[str], k: int) -> dict[str, float]:
    """Document recall and first-relevant reciprocal rank within the top-k chunks."""
    if not relevant or k < 1:
        raise ValueError('Expected nonempty gold IDs and positive k')
    ranked = document_ids[:k]
    first = next((i + 1 for i, value in enumerate(ranked) if value in relevant), None)
    return {
        'recall': len(set(ranked) & relevant) / len(relevant),
        'mrr': 1.0 / first if first else 0.0,
        'hit': float(first is not None),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--url', default='http://127.0.0.1:9000')
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--minimum-recall', type=float, default=0)
    args = parser.parse_args()
    key = os.environ.get('TEACHER_API_KEY', '')
    if not key or args.k < 1 or not 0 <= args.minimum_recall <= 1:
        parser.error('Set TEACHER_API_KEY, positive k, and minimum recall between 0 and 1')
    scores, latencies = [], []
    try:
        with args.dataset.open(encoding='utf-8') as source:
            rows = [json.loads(line) for line in source if line.strip()]
        if not rows:
            raise ValueError('Dataset must not be empty')
        with httpx.Client(
            base_url=args.url.rstrip('/'),
            headers={'Authorization': f'Bearer {key}'},
            timeout=120,
            follow_redirects=False,
        ) as client:
            for row in rows:
                gold = row['relevant_document_ids']
                if not isinstance(gold, list) or not gold or not all(isinstance(x, str) for x in gold):
                    raise ValueError('relevant_document_ids must be a nonempty string array')
                started = time.perf_counter()
                response = client.post(
                    '/api/search', json={'course_id': row['course_id'], 'query': row['query']}
                )
                response.raise_for_status()
                latencies.append(time.perf_counter() - started)
                document_ids = [hit['document_id'] for hit in response.json()['sources']]
                scores.append(score_ranking(document_ids, set(gold), args.k))
        ordered = sorted(latencies)
        result = {
            'queries': len(scores),
            'k': args.k,
            'document_recall_at_k_chunks': statistics.mean(x['recall'] for x in scores),
            'mrr_at_k_chunks': statistics.mean(x['mrr'] for x in scores),
            'hit_rate_at_k_chunks': statistics.mean(x['hit'] for x in scores),
            'retrieval_latency_p50_seconds': statistics.median(latencies),
            'retrieval_latency_p95_seconds': ordered[math.ceil(0.95 * len(ordered)) - 1],
            'note': 'Live retrieval only. Includes network/model cold starts; does not measure answer correctness.',
        }
        print(json.dumps(result, indent=2))
        return 0 if result['document_recall_at_k_chunks'] >= args.minimum_recall else 2
    except (httpx.HTTPError, ValueError, KeyError, TypeError, OSError) as exc:
        print(f'Evaluation failed ({type(exc).__name__}); no partial score published.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
