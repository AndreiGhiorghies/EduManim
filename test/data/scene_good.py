from manim import Scene, Circle, Create


class Scene1(Scene):
    def construct(self):
        circle = Circle()
        self.play(Create(circle))
        self.wait(1)