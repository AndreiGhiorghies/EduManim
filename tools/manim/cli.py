"""
tools/manim/cli.py

Command-line entry point for the Manim tool. This is the contract Track A
integrates against (Technical Plan, Section 8.2):

    python -m tools.manim.cli validate <scene.py>
        # Exit 0: valid
        # Exit 1: invalid, reason(s) printed to stderr

Day 2 scope: only `validate` is implemented here. `render` and
`render-template` land on Day 3-6.
"""

from __future__ import annotations

import argparse
import sys

from tools.manim.validator import validate


def cmd_validate(args: argparse.Namespace) -> int:
    try:
        with open(args.scene_path, "r", encoding="utf-8") as f:
            source = f.read()
    except OSError as e:
        print(f"Could not read {args.scene_path}: {e}", file=sys.stderr)
        return 1

    result = validate(source)
    if result.valid:
        print("OK: scene code is valid")
        return 0

    for error in result.errors:
        print(error, file=sys.stderr)
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m tools.manim.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate", help="Validate a Manim scene file without rendering it"
    )
    validate_parser.add_argument("scene_path", help="Path to the scene .py file")
    validate_parser.set_defaults(func=cmd_validate)

    # Placeholders so `--help` reflects the full Day 3-6 contract early;
    # these raise NotImplementedError until their days land.
    render_parser = subparsers.add_parser("render", help="Render a scene (Day 5-6)")
    render_parser.add_argument("scene_path")
    render_parser.add_argument("scene_name")
    render_parser.add_argument("--output", required=True)
    render_parser.add_argument("--quality", default="high", choices=["low", "medium", "high"])
    render_parser.set_defaults(func=_not_implemented("render"))

    template_parser = subparsers.add_parser(
        "render-template", help="Render a scene from a template (Day 3-4)"
    )
    template_parser.add_argument("template_name")
    template_parser.add_argument("--params", required=True)
    template_parser.add_argument("--output", required=True)
    template_parser.set_defaults(func=_not_implemented("render-template"))

    return parser


def _not_implemented(command: str):
    def _fn(args: argparse.Namespace) -> int:
        print(f"'{command}' is not implemented yet", file=sys.stderr)
        return 1

    return _fn


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())