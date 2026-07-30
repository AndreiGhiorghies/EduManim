import os
from manim import Scene, Circle, Create


class Scene1(Scene):
    def construct(self):
        os.system("echo pwned")
        circle = Circle()
        self.play(Create(circle))