"""Render the 40-replica gateway bundle from explicit private-network inventory."""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inventory', type=Path)
    parser.add_argument('--output', type=Path, default=Path('deploy/generated'))
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from deploy.cluster import generate_bundle

    try:
        count = generate_bundle(args.inventory, args.output)
    except (ValueError, OSError) as exc:
        # Avoid echoing private inventory or secrets accidentally placed in invalid URLs.
        print(
            f'Configuration not generated ({type(exc).__name__}); check inventory and use a new output directory.',
            file=sys.stderr,
        )
        return 1
    print(f'Generated {count} replica entries and non-secret node settings; no deployment was performed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
