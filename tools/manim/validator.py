"""Manim code validator — catches bad LLM-generated code."""
import ast
import re

ALLOWED_IMPORTS = {"from manim import *", "import manim"}

FORBIDDEN_PATTERNS = [
    r"__import__\(", r"open\(['\"]/etc", r"subprocess", r"os\.system",
    r"shutil\.", r"requests\.", r"urllib", r"socket", r"eval\(", r"exec\(",
]

def validate_manim_code(code: str) -> tuple[bool, str]:
    """Validate Manim code for safety and correctness."""
    # 1. AST parse
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    
    # 2. Forbidden patterns
    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, code):
            return False, f"Forbidden pattern: {pattern}"
    
    # 3. Import check
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not alias.name.startswith("manim"):
                    return False, f"Non-manim import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module and not node.module.startswith("manim"):
                return False, f"Non-manim import: {node.module}"
    
    # 4. Scene class check + 5. Class name pattern check
    has_valid_scene = False
    scene_class_names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                if isinstance(base, ast.Name) and base.id == "Scene":
                    scene_class_names.append(node.name)

    if not scene_class_names:
        return False, "No Scene class found"

    name_pattern = re.compile(r"^Scene\d+$")
    valid_names = [n for n in scene_class_names if name_pattern.match(n)]

    if not valid_names:
        return False, f"Scene class name(s) {scene_class_names} do not match required pattern Scene{{N}} (e.g. Scene1)"

    return True, "OK"