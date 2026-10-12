"""Create an initial private environment file with independent random credentials."""

import argparse
import os
import secrets
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('.env'))
    args = parser.parse_args()
    template = Path(__file__).resolve().parents[1] / '.env.example'
    content = template.read_text(encoding='utf-8')
    for placeholder in (
        'CHANGE_ME_ADMIN_KEY_AT_LEAST_32_CHARS',
        'CHANGE_ME_STUDENT_KEY_AT_LEAST_32_CHARS',
        'CHANGE_ME_INFERENCE_KEY',
        'CHANGE_ME_DATABASE_KEY',
    ):
        content = content.replace(placeholder, secrets.token_urlsafe(32))
    try:
        descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, 'w', encoding='utf-8') as output:
            output.write(content)
    except FileExistsError:
        print('Configuration already exists; no file was changed.', file=sys.stderr)
        return 1
    except OSError as exc:
        print(f'Configuration could not be created ({type(exc).__name__}).', file=sys.stderr)
        return 1
    print('Created private configuration with independent random credentials.')
    print('Edit service URLs and identity/course mappings; configure matching vLLM and Qdrant service keys.')
    print('No credentials were printed. Provision them through your approved secret channel.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
