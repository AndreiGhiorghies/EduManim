"""
tools/manim/templates/graph.py

Template: graph
Layout: title (top) + labeled Axes with a plotted function (center)

Params:
    title    (str, required)  - e.g. "Sine Wave"
    function (str, required)  - one of the allowed function names below.
                                 We intentionally do NOT accept an arbitrary
                                 expression string here: eval()-ing user text
                                 to build the plot would be exactly the kind
                                 of dynamic-code-execution the validator is
                                 designed to reject. Instead the LLM picks a
                                 name from a fixed, hand-verified set.
    x_range  (list[float], optional) - [min, max], default [-4, 4]
    y_range  (list[float], optional) - [min, max], default [-3, 3]
    x_label  (str, optional) - default "x"
    y_label  (str, optional) - default "f(x)"
"""

from __future__ import annotations

REQUIRED_PARAMS = ("title", "function")

# Fixed, hand-verified set of plottable functions. Extend this dict (not the
# scene code) when a new function is needed - never allow a raw expression
# string to be embedded as executable code.
ALLOWED_FUNCTIONS = {
    "sin": "np.sin(x)",
    "cos": "np.cos(x)",
    "square": "x**2",
    "cubic": "x**3",
    "linear": "x",
    "sqrt": "np.sqrt(np.abs(x))",
    "exp": "np.exp(x)",
}


def generate(params: dict, scene_name: str = "Scene1") -> str:
    missing = [p for p in REQUIRED_PARAMS if p not in params]
    if missing:
        raise ValueError(f"graph template missing required params: {missing}")

    title = str(params["title"])
    function_name = str(params["function"])
    if function_name not in ALLOWED_FUNCTIONS:
        raise ValueError(
            f"Unknown function '{function_name}'. Allowed: {sorted(ALLOWED_FUNCTIONS)}"
        )
    function_expr = ALLOWED_FUNCTIONS[function_name]

    x_range = list(params.get("x_range", [-4, 4]))
    y_range = list(params.get("y_range", [-3, 3]))
    x_label = str(params.get("x_label", "x"))
    y_label = str(params.get("y_label", "f(x)"))

    return f'''from manim import Scene, Text, Axes, Write, Create, UP
import numpy as np


class {scene_name}(Scene):
    def construct(self):
        title = Text({title!r}, font_size=40).to_edge(UP)
        self.play(Write(title), run_time=1)

        axes = Axes(
            x_range=[{x_range[0]}, {x_range[1]}, 1],
            y_range=[{y_range[0]}, {y_range[1]}, 1],
            axis_config={{"include_numbers": True}},
        )
        x_axis_label = axes.get_x_axis_label({x_label!r})
        y_axis_label = axes.get_y_axis_label({y_label!r})

        graph = axes.plot(lambda x: {function_expr}, color="#58C4DD")

        self.play(Create(axes), Write(x_axis_label), Write(y_axis_label), run_time=1.5)
        self.play(Create(graph), run_time=1.5)
        self.wait(2)
'''