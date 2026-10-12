"""Create an initial private environment file with independent random credentials."""

import argparse
import json
import os
import re
import secrets
import sys
from contextlib import ExitStack
from pathlib import Path


def create_private_files(artifacts):
    """Reserve all outputs first and roll back only our new files if any write fails."""
    created = []
    try:
        with ExitStack() as stack:
            outputs = []
            for path, content in artifacts:
                descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                info = os.fstat(descriptor)
                created.append((path, info.st_dev, info.st_ino))
                output = stack.enter_context(os.fdopen(descriptor, 'w', encoding='utf-8'))
                outputs.append((output, content))
            for output, content in outputs:
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
    except BaseException:
        for path, device, inode in created:
            try:
                info = path.lstat()
                # Never unlink a file swapped by another process after our reservation.
                if (info.st_dev, info.st_ino) == (device, inode):
                    path.unlink()
            except FileNotFoundError:
                pass
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('.env'))
    parser.add_argument(
        '--students', type=int, default=1, help='Number of independent student identities (1..256).'
    )
    parser.add_argument(
        '--course-id', default='metaverse', help='Course accessible to the generated identities.'
    )
    parser.add_argument(
        '--users-output', type=Path, help='Optional private JSON inventory for scripts/load_test.py.'
    )
    args = parser.parse_args(argv)
    if not 1 <= args.students <= 256:
        parser.error('--students must be between 1 and 256.')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', args.course_id):
        parser.error('--course-id must be a valid course identifier.')
    if args.users_output and args.output.resolve() == args.users_output.resolve():
        parser.error('--output and --users-output must be different paths.')
    template = Path(__file__).resolve().parents[1] / '.env.example'
    content = template.read_text(encoding='utf-8')
    students = [
        {
            'key': secrets.token_urlsafe(32),
            'subject': f'student-{index:03d}',
            'role': 'student',
            'courses': [args.course_id],
        }
        for index in range(1, args.students + 1)
    ]
    principals = [
        {
            'key': secrets.token_urlsafe(32),
            'subject': 'instructor',
            'role': 'admin',
            'courses': [args.course_id],
        },
        *students,
    ]
    content, replaced = re.subn(
        r'^API_KEYS=.*$',
        "API_KEYS='" + json.dumps(principals, separators=(',', ':')) + "'",
        content,
        count=1,
        flags=re.MULTILINE,
    )
    if replaced != 1:
        print('Configuration template has no API_KEYS field; no files were created.', file=sys.stderr)
        return 1
    for placeholder in ('CHANGE_ME_INFERENCE_KEY', 'CHANGE_ME_DATABASE_KEY'):
        content = content.replace(placeholder, secrets.token_urlsafe(32))
    artifacts = [(args.output, content)]
    if args.users_output:
        inventory = [{'key': student['key'], 'course_id': args.course_id} for student in students]
        artifacts.append((args.users_output, json.dumps(inventory, indent=2) + '\n'))
    try:
        create_private_files(artifacts)
    except FileExistsError:
        print(
            'An output already exists; existing files were not changed and new files were rolled back.',
            file=sys.stderr,
        )
        return 1
    except OSError as exc:
        print(f'Configuration could not be created ({type(exc).__name__}).', file=sys.stderr)
        return 1
    print('Created private configuration with independent random credentials.')
    if args.users_output:
        print(f'Created a private load-test inventory for {args.students} distinct student identities.')
    print('Edit service URLs and identity/course mappings; configure matching vLLM and Qdrant service keys.')
    print('No credentials were printed. Provision them through your approved secret channel.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
