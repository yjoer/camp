# %%
import numpy as np
from manim import Circle
from manim import Create
from manim import Cylinder
from manim import Flash
from manim import GrowFromCenter
from manim import Line
from manim import Scene
from manim import Star
from manim import ThreeDScene
from manim import VGroup


# %%
def construct_cylinder(quick=False):
    resolution = 24 if quick else [64, 96]

    cylinder = Cylinder(
        radius=0.5,
        height=1.25,
        direction=[0, 1, 0.15],
        resolution=resolution,
    )

    cylinder.rotate(angle=90 * np.pi / 180, axis=[0, 1, 0.15])
    cylinder.set_style(stroke_color="#964B00")
    cylinder.set_fill(color="#964B00", opacity=1)

    return cylinder


# %%
class CylinderScene(ThreeDScene):
    def construct(self):
        cylinder = construct_cylinder(quick=True)
        self.play(Create(cylinder))


# %manim -qh -v WARNING CylinderScene


# %%
def construct_cone(radius: int, height: int, n_lines=128):
    lines = VGroup()

    for i in range(n_lines):
        theta = 2 * np.pi * i / n_lines
        start_point = np.array([0, 0, height])
        end_point = np.array([radius * -np.cos(theta), radius * np.sin(theta), 0])

        line = Line(start=start_point, end=end_point, color="#1E792C")
        lines.add(line)

    lines.rotate(angle=90 * np.pi / 180, axis=[-1, 0, 0])
    lines.rotate(angle=180 * np.pi / 180, axis=[0, 1, 0])

    return lines


# %%
class ConeScene(ThreeDScene):
    def construct(self):
        cone = construct_cone(2, 4)
        self.play(Create(cone))


# %manim -qh -v WARNING ConeScene


# %%
def sample_points_from_triangles(vertices: list[tuple[float, float]], n_samples: int):
    rng = np.random.default_rng(seed=26)
    t1 = rng.random(size=n_samples)
    t2 = rng.random(size=n_samples)

    mask = t1 + t2 > 1
    t1[mask] = 1 - t1[mask]
    t2[mask] = 1 - t2[mask]
    t3 = 1 - t1 - t2

    x = t1 * vertices[0][0] + t2 * vertices[1][0] + t3 * vertices[2][0]
    y = t1 * vertices[0][1] + t2 * vertices[1][1] + t3 * vertices[2][1]

    return x, y


def construct_dots(vertices: list[tuple[float, float]], n_dots: int):
    x, y = sample_points_from_triangles(vertices, n_samples=n_dots)

    palette = ["#FA4032", "#FF8383", "#FFC145", "#A294F9", "#B1F0F7"]
    rng = np.random.default_rng(seed=26)
    colors = rng.choice(palette, size=len(x))

    circles = VGroup()

    for i in range(n_dots):
        circle = Circle(radius=0.05)
        circle.set_style(stroke_width=0)
        circle.set_fill(color=colors[i], opacity=1)
        circle.shift([x[i], y[i], 0])

        circles.add(circle)

    return circles


# %%
class DotScene(Scene):
    def construct(self):
        cone = construct_cone(2, 4)
        self.play(Create(cone))

        vertices = [(0, 2), (-2, -2), (2, -2)]
        dots = construct_dots(vertices, n_dots=75)
        self.play(Create(dots))


# %manim -qh -v WARNING DotScene


# %%
def construct_star():
    star = Star(n=6, outer_radius=0.5, inner_radius=0.25)
    star.set_style(stroke_width=0)
    star.set_fill(color="#FFDF00", opacity=1)

    return star


# %%
class StarScene(Scene):
    def construct(self):
        star = construct_star()
        self.play(GrowFromCenter(star), run_time=0.5)
        self.play(
            Flash(
                star,
                line_length=0.5,
                flash_radius=0.55,
                line_stroke_width=4,
                color="#FFDF00",
            )
        )


# %manim -qh -v WARNING StarScene


# %%
class TreeScene(ThreeDScene):
    def construct(self):
        trunk = construct_cylinder(quick=False)
        trunk.shift([0, -2.5, 0])
        self.play(Create(trunk, run_time=0.5))

        leaves = construct_cone(2, 4)
        self.play(Create(leaves, run_time=5))

        vertices = [(0, 2), (-2, -2), (2, -2)]
        dots = construct_dots(vertices, n_dots=75)
        self.play(Create(dots))

        star = construct_star()
        star.shift([0, 2.25, 0])
        self.play(GrowFromCenter(star), run_time=0.5)
        self.play(
            Flash(
                star,
                line_length=0.5,
                flash_radius=0.55,
                line_stroke_width=4,
                color="#FFDF00",
            )
        )

        self.wait()


# %manim -qh -v WARNING TreeScene

# %%
