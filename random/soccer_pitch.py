# %%
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from ellipse import LsqEllipse
from IPython.display import display
from matplotlib.patches import Ellipse

from solutions.sports.yolo_v8_calibration_pipeline import SoccerPitch
from solutions.sports.yolo_v8_calibration_pipeline import (
    find_ellipse_tangent_intersections,
)

# %matplotlib inline
# %config InlineBackend.figure_formats = ['retina']

# %load_ext autoreload
# %autoreload 2

# %% language="html"
#
# <style>
#     mjx-assistive-mml {
#         height: 100%;
#     }
# </style>

# %%
pitch = SoccerPitch()

# %%
fig = pitch.plot()
fig.show()


# %%
def make_ellipse(
    n_samples: int,
    width: int,
    height: int,
    theta: int,
    noise: float,
    random_state: Optional[int],
):
    t = np.linspace(0, 2 * np.pi, n_samples)
    x = width * np.cos(t)
    y = height * np.sin(t)
    theta = np.radians(theta)

    rng = np.random.default_rng(seed=random_state)
    gaussian_noise = rng.normal(size=(len(t), 2))

    rotation_matrix = np.array(
        [
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)],
        ]
    )

    points = (np.vstack([x, y]).T @ rotation_matrix) + (noise * gaussian_noise)

    return points[:, 0], points[:, 1]


# %%
width, height, theta = 24, 16, 30
x, y = make_ellipse(250, width, height, theta, noise=3, random_state=26)

reg = LsqEllipse().fit(np.array([x, y]).T)
center, width, height, theta = reg.as_parameters()

ellipse_kwargs = {
    "xy": center,
    "width": width * 2,
    "height": height * 2,
    "angle": np.degrees(theta),
    "edgecolor": "orange",
    "fc": "None",
    "label": "Least Squares Fit",
}

print(center, width, height, np.degrees(theta))

plt.scatter(x, y, label="Points")
plt.gca().add_patch(Ellipse(**ellipse_kwargs))
plt.legend()
plt.show()

# %%
symbols = ["a", "b", "c", "d", "e", "f"]
grid = np.empty((len(symbols), len(symbols)), dtype="U2")

for i, v1 in enumerate(symbols):
    for j, v2 in enumerate(symbols):
        grid[i][j] = f"{v1}{v2}"

grid

# %% [markdown]
# $(a + b + c + d + e + f)^2$ \
# $= a^2 + b^2 + c^2 + d^2 + e^2 + f^2$ \
# $+ 2ab + 2ac + 2ad + 2ae + 2af$ \
# $+ 2bc + 2bd + 2be + 2bf$ \
# $+ 2cd + 2ce + 2cf$ \
# $+ 2de + 2df$ \
# $+ 2ef$

# %%
symbols = ["a", "b", "c", "d", "e", "f", "g", "h"]
grid = np.empty((len(symbols), len(symbols)), dtype="U2")

for i, v1 in enumerate(symbols):
    for j, v2 in enumerate(symbols):
        grid[i][j] = f"{v1}{v2}"

grid

# %% [markdown]
# $(a + b + c + d + e + f + g + h)^2$ \
# $= a^2 + b^2 + c^2 + d^2 + e^2 + f^2 + g^2 + h^2$ \
# $+ 2ab + 2ac + 2ad + 2ae + 2af + 2ag + 2ah$ \
# $+ 2bc + 2bd + 2be + 2bf + 2bg + 2bh$ \
# $+ 2cd + 2ce + 2cf + 2cg + 2ch$ \
# $+ 2de + 2df + 2dg + 2dh$ \
# $+ 2ef + 2eg + 2eh$ \
# $+ 2fg + 2fh$ \
# $+ 2gh$

# %% [markdown]
# $$
# \begin{flalign}
# Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0 \tag{1} &&
# \end{flalign}
# $$
#
# $$
# \begin{flalign}
# y - y_1 &= m(x - x_1) \\
# y &= y_1 + mx - mx_1 &&
# \tag{2}
# \end{flalign}
# $$
#
# Substitute $(2)$ into $(1)$ to eliminate $y:$
#
# $Ax^2 + Bx(y_1 + mx - mx_1) + C(y_1 + mx - mx_1)^2 + Dx + E(y_1 + mx - mx_1) + F = 0$
#
# Expand:
#
# $$
# \begin{flalign}
# Ax^2 &+ Bmx^2 - Bmxx_1 + Bxy_1 \\
# &+ Cm^2x^2 + Cm^2x_1^2 - 2Cm^2xx_1 + 2Cmxy_1 - 2Cmx_1y_1 + Cy_1^2 \\
# &+ Dx + Emx - Emx_1 + Ey_1 + F = 0 &&
# \tag{3}
# \end{flalign}
# $$
#
# Rearrange $(3)$ into $A'x^2 + B'x + C' = 0:$
#
# $A' = A + Bm + Cm^2$ \
# $B' = -Bmx_1 + By_1 - 2Cm^2x_1 + 2Cmy_1 + D + Em$ \
# $C' = Cm^2x_1^2 - 2Cmx_1y_1 + Cy_1^2 - Emx_1 + Ey_1 + F$

# %% [markdown]
# Discriminant at the tangent: $B'^2 - 4A'C' = 0$
#
# $$
# \begin{flalign}
# &B'^2 \\
# &= B^2m^2x_1^2 + B^2y_1^2 + 4C^2m^4x_1^2 + 4C^2m^2y_1^2 + D^2 + E^2m^2 \\
# &- 2B^2mx_1y_1 + 4BCm^3x_1^2 - 4BCm^2x_1y_1 - 2BDmx_1 - 2BEm^2x_1 \\
# &- 4BCm^2x_1y_1 + 4BCmy_1^2 + 2BDy_1 + 2BEmy_1 \\
# &- 8C^2m^3x_1y_1 - 4CDm^2x_1 - 4CEm^3x_1 \\
# &+ 4CDmy_1 + 4CEm^2y_1 \\
# &+ 2DEm &&
# \end{flalign}
# $$
#
# $$
# \begin{flalign}
# &4A'C' \\
# &= 4ACm^2x_1^2 - 8ACmx_1y_1 + 4ACy_1^2 - 4AEmx_1 + 4AEy_1 + 4AF \\
# &+ 4BCm^3x_1^2 - 8BCm^2x_1y_1 + 4BCmy_1^2 - 4BEm^2x_1 + 4BEmy_1 + 4BFm \\
# &+ 4C^2m^4x_1^2 - 8C^2m^3x_1y_1 + 4C^2m^2y_1^2 - 4CEm^3x_1+ 4CEm^2y_1 + 4CFm^2 &&
# \end{flalign}
# $$
#
# $$
# \begin{flalign}
# &B'^2 - 4A'C' \\
# &= B^2m^2x_1^2 + B^2y_1^2 + D^2 + E^2m^2 - 2B^2mx_1y_1 - 2BDmx_1 \\
# &+ 2BEm^2x_1 + 2BDy_1 - 2BEmy_1 - 4CDm^2x_1 + 4CDmy_1 + 2DEm \\
# &- 4ACm^2x_1^2 + 8ACmx_1y_1 - 4ACy_1^2 + 4AEmx_1 - 4AEy_1 - 4AF - 4BFm - 4CFm^2 &&
# \tag{4}
# \end{flalign}
# $$
#
# Rearrange $(4)$ into $A''m^2 + B''m + C'' = 0:$
#
# $A'' = B^2x_1^2 + E^2 + 2BEx_1 - 4CDx_1 - 4ACx_1^2 - 4CF$ \
# $B'' = -2B^2x_1y_1 - 2BDx_1 - 2BEy_1 + 4CDy_1 + 2DE + 8ACx_1y_1 + 4AEx_1 - 4BF$ \
# $C'' = B^2y_1^2 + D^2 + 2BDy_1 - 4 ACy_1^2 - 4AEy_1 - 4AF$

# %% [markdown]
# Two Distinct Roots of Quadratic Equations:
#
# $$
# \begin{flalign}
# x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a} &&
# \end{flalign}
# $$
#
# Equal Roots of Quadratic Equations:
#
# $$
# \begin{flalign}
# \because \; &b^2 - 4ac = 0 \text{ and } \sqrt{0} = 0 \\
# &x = \frac{-b}{2a} &&
# \end{flalign}
# $$

# %%
A, B, C, D, E, F = sp.symbols("A B C D E F")
m, x, y, x1, y1 = sp.symbols("m x y x1 y1")

# %%
ellipse_eq = A * x**2 + B * x * y + C * y**2 + D * x + E * y + F
line_eq = y1 + m * x - m * x1

ellipse_eq_y = sp.expand(ellipse_eq.subs(y, line_eq))
display(ellipse_eq_y)

# %%
A1 = ellipse_eq_y.coeff(x, 2)
B1 = ellipse_eq_y.coeff(x, 1)
C1 = ellipse_eq_y.coeff(x, 0)

display(A1)
display(B1)
display(C1)

# %%
discriminant = sp.expand(B1**2 - 4 * A1 * C1)

A2 = discriminant.coeff(m, 2)
B2 = discriminant.coeff(m, 1)
C2 = discriminant.coeff(m, 0)

display(A2)
display(B2)
display(C2)

# %%
root_m = sp.simplify((-B2 + sp.sqrt(sp.factor(B2**2 - 4 * A2 * C2))) / (2 * A2))
root_m

# %%
root_x = sp.factor(-B1 / (2 * A1))
root_x

# %%
point = np.array([-5, 50])
points = find_ellipse_tangent_intersections(reg.coefficients, point)

plt.plot([point[0], points[0][0]], [point[1], points[0][1]], zorder=2)
plt.plot([point[0], points[1][0]], [point[1], points[1][1]], zorder=2)

plt.scatter(point[0], point[1], label="Point", zorder=3)

plt.scatter(
    [points[0][0], points[1][0]],
    [points[0][1], points[1][1]],
    label="Intersections",
    zorder=3,
)

plt.gca().add_patch(Ellipse(**ellipse_kwargs))
plt.legend()
plt.show()
