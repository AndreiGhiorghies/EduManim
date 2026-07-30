"""
tests/unit/test_validator.py

Unit tests for tools/manim/validator.py.

Covers the Day 2 deliverable: "Validator catches 5+ types of bad Manim
code." Each `test_bad_*` below is a distinct failure mode.
"""

import pytest

from tools.manim.validator import validate


# --- Good path -----------------------------------------------------------


def test_good_scene_is_valid():
    source = """
from manim import Scene, Circle, Create

class Scene1(Scene):
    def construct(self):
        circle = Circle()
        self.play(Create(circle))
"""
    result = validate(source)
    assert result.valid
    assert result.errors == []


# --- Bad code sample 1: syntax error --------------------------------------


def test_bad_syntax_error():
    source = """
from manim import Scene

class Scene1(Scene)
    def construct(self):
        pass
"""
    result = validate(source)
    assert not result.valid
    assert any("SyntaxError" in e for e in result.errors)


# --- Bad code sample 2: disallowed import ----------------------------------


def test_bad_disallowed_import():
    source = """
import os
from manim import Scene

class Scene1(Scene):
    def construct(self):
        pass
"""
    result = validate(source)
    assert not result.valid
    assert any("import of 'os'" in e for e in result.errors)


def test_bad_disallowed_import_from():
    source = """
from subprocess import run
from manim import Scene

class Scene1(Scene):
    def construct(self):
        pass
"""
    result = validate(source)
    assert not result.valid
    assert any("import from 'subprocess'" in e for e in result.errors)


# --- Bad code sample 3: forbidden calls (eval/exec/os.system/subprocess) --


def test_bad_eval_call():
    source = """
from manim import Scene

class Scene1(Scene):
    def construct(self):
        eval("1 + 1")
"""
    result = validate(source)
    assert not result.valid
    assert any("forbidden call 'eval" in e for e in result.errors)


def test_bad_os_system_call():
    source = """
import os
from manim import Scene

class Scene1(Scene):
    def construct(self):
        os.system("rm -rf /")
"""
    result = validate(source)
    assert not result.valid
    assert any("forbidden call 'os.system" in e for e in result.errors)


def test_bad_subprocess_call():
    source = """
import subprocess
from manim import Scene

class Scene1(Scene):
    def construct(self):
        subprocess.run(["ls"])
"""
    result = validate(source)
    assert not result.valid
    assert any("forbidden call 'subprocess.run" in e for e in result.errors)


def test_bad_file_write_outside_sandbox():
    source = """
from manim import Scene

class Scene1(Scene):
    def construct(self):
        f = open("/etc/passwd", "w")
        f.write("oops")
"""
    result = validate(source)
    assert not result.valid
    assert any("file operations are not allowed" in e for e in result.errors)


def test_bad_network_call():
    source = """
import requests
from manim import Scene

class Scene1(Scene):
    def construct(self):
        requests.get("http://example.com")
"""
    result = validate(source)
    assert not result.valid
    assert any("import of 'requests'" in e for e in result.errors)


# --- Bad code sample 4: no Scene subclass at all --------------------------


def test_bad_no_scene_class():
    source = """
from manim import Circle

def construct():
    circle = Circle()
"""
    result = validate(source)
    assert not result.valid
    assert any("No class inheriting from 'Scene'" in e for e in result.errors)


# --- Bad code sample 5: wrong class name pattern --------------------------


def test_bad_wrong_class_name():
    source = """
from manim import Scene, Circle, Create

class MyCoolScene(Scene):
    def construct(self):
        circle = Circle()
        self.play(Create(circle))
"""
    result = validate(source)
    assert not result.valid
    assert any("must match the pattern 'Scene<N>'" in e for e in result.errors)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))