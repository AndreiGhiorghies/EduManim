"""
tools/manim/validator.py

Static validator for LLM-generated Manim scene code.

Scene code is produced by an LLM, so it can never be trusted to be
well-formed or safe. This module runs several independent checks
BEFORE the code is ever executed or handed to `manim` for rendering:

    1. AST parse check        - is it even syntactically valid Python?
    2. Import allowlist        - only `manim` may be imported
    3. Forbidden pattern check - no os.system / eval / exec / subprocess /
                                 network calls / file ops
    4. Scene class detection   - at least one class must inherit from `Scene`
    5. Class name pattern      - that class must be named `Scene<N>` (e.g. Scene1)

Any single failure makes the code INVALID. Validation never executes the
code being checked - everything is done via `ast` inspection only.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Optional

# --- Configuration -----------------------------------------------------

# Only these top-level module names may be imported. Extend cautiously -
# every entry here is trusted to have no dangerous side effects on import.
ALLOWED_IMPORT_MODULES = {
    "manim",
}

# Bare names that must never appear as a Call target anywhere in the tree.
# Covers dynamic code execution and dynamic import.
FORBIDDEN_CALL_NAMES = {
    "eval",
    "exec",
    "compile",
    "__import__",
}

# Dotted-name prefixes that indicate process spawning, network access, or
# other capabilities scene code has no legitimate reason to use. Matched
# against both attribute-call targets (os.system(...)) and bare attribute
# references (os.environ).
FORBIDDEN_MODULE_PREFIXES = (
    "os",
    "sys",
    "subprocess",
    "socket",
    "shutil",
    "urllib",
    "requests",
    "http",
    "ftplib",
    "smtplib",
    "importlib",
    "ctypes",
    "multiprocessing",
    "threading",
    "pickle",
    "marshal",
)

SCENE_NAME_PATTERN = re.compile(r"^Scene\d+$")


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.valid


def validate(source: str) -> ValidationResult:
    """Run all checks against a scene source string and return the result."""
    # 1. AST parse check ---------------------------------------------------
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return ValidationResult(False, [f"SyntaxError: {e.msg} (line {e.lineno})"])

    errors: list[str] = []
    errors += _check_imports(tree)          # 2. Import allowlist
    errors += _check_forbidden_patterns(tree)  # 3. Forbidden patterns
    errors += _check_scene_class(tree)       # 4 & 5. Scene class + naming

    return ValidationResult(len(errors) == 0, errors)


def validate_file(path: str) -> ValidationResult:
    """Convenience wrapper: read a file from disk and validate its contents."""
    with open(path, "r", encoding="utf-8") as f:
        return validate(f.read())


# --- Internal check implementations -------------------------------------


def _root_module(name: str) -> str:
    return name.split(".")[0]


def _check_imports(tree: ast.AST) -> list[str]:
    errors = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = _root_module(alias.name)
                if root not in ALLOWED_IMPORT_MODULES:
                    errors.append(
                        f"Line {node.lineno}: import of '{alias.name}' is not allowed "
                        f"(only {sorted(ALLOWED_IMPORT_MODULES)} permitted)"
                    )
        elif isinstance(node, ast.ImportFrom):
            root = _root_module(node.module or "")
            if root not in ALLOWED_IMPORT_MODULES:
                errors.append(
                    f"Line {node.lineno}: import from '{node.module}' is not allowed "
                    f"(only {sorted(ALLOWED_IMPORT_MODULES)} permitted)"
                )
    return errors


def _dotted_name(node: ast.Attribute) -> Optional[str]:
    """Reconstruct a dotted name like 'os.path.join' from an Attribute node."""
    parts = []
    cur: ast.AST = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return ".".join(reversed(parts))
    return None


def _check_forbidden_patterns(tree: ast.AST) -> list[str]:
    errors = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = None
            dotted = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
                dotted = _dotted_name(func)

            if name in FORBIDDEN_CALL_NAMES:
                errors.append(f"Line {node.lineno}: forbidden call '{name}(...)'")

            if dotted and dotted.split(".")[0] in FORBIDDEN_MODULE_PREFIXES:
                errors.append(f"Line {node.lineno}: forbidden call '{dotted}(...)'")

            if name == "open":
                errors.append(
                    f"Line {node.lineno}: file operations are not allowed in scene code"
                )

        # Bare references to forbidden modules (e.g. `os.environ`) even
        # without a call, since attribute access alone can leak data.
        if isinstance(node, ast.Attribute):
            dotted = _dotted_name(node)
            if dotted and dotted.split(".")[0] in FORBIDDEN_MODULE_PREFIXES:
                errors.append(f"Line {node.lineno}: reference to forbidden module '{dotted}'")

    return errors


def _check_scene_class(tree: ast.AST) -> list[str]:
    errors = []
    scene_classes = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            base_names = [
                b.id if isinstance(b, ast.Name) else getattr(b, "attr", None)
                for b in node.bases
            ]
            if "Scene" in base_names:
                scene_classes.append(node)

    if not scene_classes:
        errors.append("No class inheriting from 'Scene' was found")
        return errors

    for cls in scene_classes:
        if not SCENE_NAME_PATTERN.match(cls.name):
            errors.append(
                f"Line {cls.lineno}: class '{cls.name}' inherits from Scene but its "
                f"name must match the pattern 'Scene<N>' (e.g. Scene1)"
            )

    return errors