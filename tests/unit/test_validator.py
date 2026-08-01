from tools.manim.validator import validate_manim_code, validate_manim_file


def test_good_scene_passes():
    result = validate_manim_file("test/data/scene_good.py")
    assert result.ok
    assert result.scene_class_name == "Scene1"
    assert result.errors == []


def test_bad_scene_fails_on_os_system():
    result = validate_manim_file("test/data/scene_bad.py")
    assert not result.ok
    assert any("Forbidden pattern" in e for e in result.errors)


def test_empty_code_fails():
    result = validate_manim_code("")
    assert not result.ok
    assert "Code is empty" in result.errors[0]


def test_syntax_error_fails():
    result = validate_manim_code("def broken(:\n    pass")
    assert not result.ok
    assert "Syntax error" in result.errors[0]


def test_non_manim_import_fails():
    code = """import os
from manim import Scene
class Scene1(Scene):
    def construct(self):
        pass
"""
    result = validate_manim_code(code)
    assert not result.ok
    assert any("Non-manim import" in e for e in result.errors)


def test_no_scene_class_fails():
    code = """from manim import Scene
class NotAScene:
    pass
"""
    result = validate_manim_code(code)
    assert not result.ok
    assert any("No class inheriting from Scene" in e for e in result.errors)


def test_multiple_scene_classes_fails():
    code = """from manim import Scene
class Scene1(Scene):
    def construct(self):
        pass
class Scene2(Scene):
    def construct(self):
        pass
"""
    result = validate_manim_code(code)
    assert not result.ok
    assert any("Expected exactly one Scene subclass" in e for e in result.errors)


def test_wrong_scene_name_pattern_fails():
    code = """from manim import Scene
class MyScene(Scene):
    def construct(self):
        pass
"""
    result = validate_manim_code(code)
    assert not result.ok
    assert any("does not match required pattern" in e for e in result.errors)


def test_expected_scene_id_mismatch_fails():
    code = """from manim import Scene
class Scene1(Scene):
    def construct(self):
        pass
"""
    result = validate_manim_code(code, expected_scene_id=2)
    assert not result.ok
    assert any("expected 'Scene2'" in e for e in result.errors)


def test_eval_and_exec_are_forbidden():
    code_eval = """from manim import Scene
class Scene1(Scene):
    def construct(self):
        eval("1+1")
"""
    result = validate_manim_code(code_eval)
    assert not result.ok

    code_exec = """from manim import Scene
class Scene1(Scene):
    def construct(self):
        exec("x=1")
"""
    result = validate_manim_code(code_exec)
    assert not result.ok


def test_missing_file_reports_as_validation_failure():
    result = validate_manim_file("test/data/does_not_exist.py")
    assert not result.ok
    assert "Could not read file" in result.errors[0]
