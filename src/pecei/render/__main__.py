"""CLI: ``python -m pecei.render <map.yaml> [...]`` — preview one or more maps."""
from __future__ import annotations

import sys

from pecei.world.map_parser import load_map_spec

from .preview import show


def main(argv: list[str] | None = None) -> None:
    args = list(argv if argv is not None else sys.argv[1:])
    if not args:
        print("usage: python -m pecei.render <map.yaml> [more maps...]", file=sys.stderr)
        raise SystemExit(2)
    for path in args:
        spec = load_map_spec(path)
        show(spec.build_grid(), spec.build_entities(), title=f"PECEI — {path}")


if __name__ == "__main__":
    main()
