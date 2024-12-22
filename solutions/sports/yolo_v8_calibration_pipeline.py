from typing import Generic
from typing import Literal
from typing import TypedDict
from typing import TypeVar

import numpy as np
import plotly.graph_objects as go
import torch
import torchvision.transforms.v2.functional as tvf
from ellipse import LsqEllipse
from numpy.polynomial import polynomial as P


def transforms(image, target):
    image = tvf.to_image(image)
    image = tvf.to_dtype(image, dtype=torch.float32, scale=True)

    return image, target


def find_circle_tangent_intersections(
    circle: tuple[float, float],
    radius: float,
    point: tuple[float, float],
):
    cx, cy = circle
    px, py = point
    dx, dy = px - cx, py - cy

    hypotenuse = np.sqrt(dx**2 + dy**2)
    theta = np.arccos(radius / hypotenuse)

    d = np.arctan2(dy, dx)
    d1 = d - theta
    d2 = d + theta

    intersection_1 = np.array(
        [cx + radius * np.cos(d1), cy + radius * np.sin(d1), 0],
        dtype=np.float32,
    )

    intersection_2 = np.array(
        [cx + radius * np.cos(d2), cy + radius * np.sin(d2), 0],
        dtype=np.float32,
    )

    return intersection_1, intersection_2


# https://docs.python.org/3/whatsnew/3.12.html#pep-695-type-parameter-syntax
T = TypeVar("T")


class SoccerPitchKeypoints(TypedDict, Generic[T]):
    center_mark: T
    halfway_top_touch_line_mark: T
    halfway_bottom_touch_line_mark: T
    halfway_line_center_circle_top_mark: T
    halfway_line_center_circle_bottom_mark: T
    top_left_corner: T
    top_right_corner: T
    bottom_left_corner: T
    bottom_right_corner: T
    left_goal_top_left_post: T
    left_goal_top_right_post: T
    left_goal_bottom_left_post: T
    left_goal_bottom_right_post: T
    right_goal_top_left_post: T
    right_goal_top_right_post: T
    right_goal_bottom_left_post: T
    right_goal_bottom_right_post: T
    left_goal_area_top_left_corner: T
    left_goal_area_top_right_corner: T
    left_goal_area_bottom_left_corner: T
    left_goal_area_bottom_right_corner: T
    right_goal_area_top_left_corner: T
    right_goal_area_top_right_corner: T
    right_goal_area_bottom_left_corner: T
    right_goal_area_bottom_right_corner: T
    left_penalty_mark: T
    right_penalty_mark: T
    left_penalty_area_top_left_corner: T
    left_penalty_area_top_right_corner: T
    left_penalty_area_bottom_left_corner: T
    left_penalty_area_bottom_right_corner: T
    right_penalty_area_top_left_corner: T
    right_penalty_area_top_right_corner: T
    right_penalty_area_bottom_left_corner: T
    right_penalty_area_bottom_right_corner: T
    top_left_penalty_ark_mark: T
    top_right_penalty_ark_mark: T
    bottom_left_penalty_ark_mark: T
    bottom_right_penalty_ark_mark: T
    center_circle_top_left_corner: T
    center_circle_top_right_corner: T
    center_circle_bottom_left_corner: T
    center_circle_bottom_right_corner: T
    center_circle_left: T
    center_circle_right: T
    left_penalty_ark_mark: T
    left_penalty_ark_right_mark: T
    right_penalty_ark_mark: T
    right_penalty_ark_left_mark: T
    center_circle_top_left_tangent: T
    center_circle_top_right_tangent: T
    center_circle_bottom_left_tangent: T
    center_circle_bottom_right_tangent: T
    left_penalty_ark_top_tangent: T
    left_penalty_ark_bottom_tangent: T
    right_penalty_ark_top_tangent: T
    right_penalty_ark_bottom_tangent: T


class SoccerPitch:
    PITCH_LENGTH = 105.0
    PITCH_WIDTH = 68.0

    CENTER_CIRCLE_RADIUS = 9.15

    PENALTY_AREA_LENGTH = 16.5
    PENALTY_AREA_WIDTH = 40.32

    GOAL_AREA_LENGTH = 5.5
    GOAL_AREA_WIDTH = 18.32

    GOAL_LINE_TO_PENALTY_MARK = 11.0
    GOAL_WIDTH = 7.32
    GOAL_HEIGHT = 2.44

    def __init__(self):
        lkp_1 = [-self.PITCH_LENGTH / 2 + self.GOAL_AREA_LENGTH, -self.GOAL_AREA_WIDTH / 2, 0]  # fmt: skip
        lkp_2 = [-self.PITCH_LENGTH / 2 + self.GOAL_AREA_LENGTH, self.GOAL_AREA_WIDTH / 2, 0]  # fmt: skip

        rkp_1 = [self.PITCH_LENGTH / 2 - self.GOAL_AREA_LENGTH, -self.GOAL_AREA_WIDTH / 2, 0]  # fmt: skip
        rkp_2 = [self.PITCH_LENGTH / 2 - self.GOAL_AREA_LENGTH, self.GOAL_AREA_WIDTH / 2, 0]  # fmt: skip

        lkp_3 = [-self.PITCH_LENGTH / 2 + self.PENALTY_AREA_LENGTH, -self.PENALTY_AREA_WIDTH / 2, 0]  # fmt: skip
        lkp_4 = [-self.PITCH_LENGTH / 2 + self.PENALTY_AREA_LENGTH, self.PENALTY_AREA_WIDTH / 2, 0]  # fmt: skip

        rkp_3 = [self.PITCH_LENGTH / 2 - self.PENALTY_AREA_LENGTH, -self.PENALTY_AREA_WIDTH / 2, 0]  # fmt: skip
        rkp_4 = [self.PITCH_LENGTH / 2 - self.PENALTY_AREA_LENGTH, self.PENALTY_AREA_WIDTH / 2, 0]  # fmt: skip

        self.keypoints: SoccerPitchKeypoints[np.ndarray[tuple[Literal[2]]]] = {
            "center_mark": np.array(
                [0, 0, 0],
                dtype=np.float32,
            ),
            "halfway_top_touch_line_mark": np.array(
                [0, -self.PITCH_WIDTH / 2, 0],
                dtype=np.float32,
            ),
            "halfway_bottom_touch_line_mark": np.array(
                [0, self.PITCH_WIDTH / 2, 0],
                dtype=np.float32,
            ),
            "halfway_line_center_circle_top_mark": np.array(
                [0, -self.CENTER_CIRCLE_RADIUS, 0],
                dtype=np.float32,
            ),
            "halfway_line_center_circle_bottom_mark": np.array(
                [0, self.CENTER_CIRCLE_RADIUS, 0],
                dtype=np.float32,
            ),
            "top_left_corner": np.array(
                [-self.PITCH_LENGTH / 2, -self.PITCH_WIDTH / 2, 0],
                dtype=np.float32,
            ),
            "top_right_corner": np.array(
                [self.PITCH_LENGTH / 2, -self.PITCH_WIDTH / 2, 0],
                dtype=np.float32,
            ),
            "bottom_left_corner": np.array(
                [-self.PITCH_LENGTH / 2, self.PITCH_WIDTH / 2, 0],
                dtype=np.float32,
            ),
            "bottom_right_corner": np.array(
                [self.PITCH_LENGTH / 2, self.PITCH_WIDTH / 2, 0],
                dtype=np.float32,
            ),
            "left_goal_top_left_post": np.array(
                [-self.PITCH_LENGTH / 2, self.GOAL_WIDTH / 2, -self.GOAL_HEIGHT],
                dtype=np.float32,
            ),
            "left_goal_top_right_post": np.array(
                [-self.PITCH_LENGTH / 2, -self.GOAL_WIDTH / 2, -self.GOAL_HEIGHT],
                dtype=np.float32,
            ),
            "left_goal_bottom_left_post": np.array(
                [-self.PITCH_LENGTH / 2, self.GOAL_WIDTH / 2, 0],
                dtype=np.float32,
            ),
            "left_goal_bottom_right_post": np.array(
                [-self.PITCH_LENGTH / 2, -self.GOAL_WIDTH / 2, 0],
                dtype=np.float32,
            ),
            "right_goal_top_left_post": np.array(
                [self.PITCH_LENGTH / 2, -self.GOAL_WIDTH / 2, -self.GOAL_HEIGHT],
                dtype=np.float32,
            ),
            "right_goal_top_right_post": np.array(
                [self.PITCH_LENGTH / 2, self.GOAL_WIDTH / 2, -self.GOAL_HEIGHT],
                dtype=np.float32,
            ),
            "right_goal_bottom_left_post": np.array(
                [self.PITCH_LENGTH / 2, -self.GOAL_WIDTH / 2, 0],
                dtype=np.float32,
            ),
            "right_goal_bottom_right_post": np.array(
                [self.PITCH_LENGTH / 2, self.GOAL_WIDTH / 2, 0],
                dtype=np.float32,
            ),
            "left_goal_area_top_left_corner": np.array(
                [-self.PITCH_LENGTH / 2, -self.GOAL_AREA_WIDTH / 2, 0],
                dtype=np.float32,
            ),
            "left_goal_area_top_right_corner": np.array(
                lkp_1,
                dtype=np.float32,
            ),
            "left_goal_area_bottom_left_corner": np.array(
                [-self.PITCH_LENGTH / 2, self.GOAL_AREA_WIDTH / 2, 0],
                dtype=np.float32,
            ),
            "left_goal_area_bottom_right_corner": np.array(
                lkp_2,
                dtype=np.float32,
            ),
            "right_goal_area_top_left_corner": np.array(
                rkp_1,
                dtype=np.float32,
            ),
            "right_goal_area_top_right_corner": np.array(
                [self.PITCH_LENGTH / 2, -self.GOAL_AREA_WIDTH / 2, 0],
                dtype=np.float32,
            ),
            "right_goal_area_bottom_left_corner": np.array(
                rkp_2,
                dtype=np.float32,
            ),
            "right_goal_area_bottom_right_corner": np.array(
                [self.PITCH_LENGTH / 2, self.GOAL_AREA_WIDTH / 2, 0],
                dtype=np.float32,
            ),
            "left_penalty_mark": np.array(
                [-self.PITCH_LENGTH / 2 + self.GOAL_LINE_TO_PENALTY_MARK, 0, 0],
                dtype=np.float32,
            ),
            "right_penalty_mark": np.array(
                [self.PITCH_LENGTH / 2 - self.GOAL_LINE_TO_PENALTY_MARK, 0, 0],
                dtype=np.float32,
            ),
            "left_penalty_area_top_left_corner": np.array(
                [-self.PITCH_LENGTH / 2, -self.PENALTY_AREA_WIDTH / 2, 0],
                dtype=np.float32,
            ),
            "left_penalty_area_top_right_corner": np.array(
                lkp_3,
                dtype=np.float32,
            ),
            "left_penalty_area_bottom_left_corner": np.array(
                [-self.PITCH_LENGTH / 2, self.PENALTY_AREA_WIDTH / 2, 0],
                dtype=np.float32,
            ),
            "left_penalty_area_bottom_right_corner": np.array(
                lkp_4,
                dtype=np.float32,
            ),
            "right_penalty_area_top_left_corner": np.array(
                rkp_3,
                dtype=np.float32,
            ),
            "right_penalty_area_top_right_corner": np.array(
                [self.PITCH_LENGTH / 2, -self.PENALTY_AREA_WIDTH / 2, 0],
                dtype=np.float32,
            ),
            "right_penalty_area_bottom_left_corner": np.array(
                rkp_4,
                dtype=np.float32,
            ),
            "right_penalty_area_bottom_right_corner": np.array(
                [self.PITCH_LENGTH / 2, self.PENALTY_AREA_WIDTH / 2, 0],
                dtype=np.float32,
            ),
        }

        # Use the Pythagorean theorem to calculate the intersection point between the
        # penalty area and the ark.
        dx = self.PENALTY_AREA_LENGTH - self.GOAL_LINE_TO_PENALTY_MARK
        dy = np.sqrt(self.CENTER_CIRCLE_RADIUS**2 - dx**2)

        self.keypoints["top_left_penalty_ark_mark"] = np.array(
            [-self.PITCH_LENGTH / 2 + self.PENALTY_AREA_LENGTH, -dy, 0],
            dtype=np.float32,
        )

        self.keypoints["top_right_penalty_ark_mark"] = np.array(
            [self.PITCH_LENGTH / 2 - self.PENALTY_AREA_LENGTH, -dy, 0],
            dtype=np.float32,
        )

        self.keypoints["bottom_left_penalty_ark_mark"] = np.array(
            [-self.PITCH_LENGTH / 2 + self.PENALTY_AREA_LENGTH, dy, 0],
            dtype=np.float32,
        )

        self.keypoints["bottom_right_penalty_ark_mark"] = np.array(
            [self.PITCH_LENGTH / 2 - self.PENALTY_AREA_LENGTH, dy, 0],
            dtype=np.float32,
        )

        # x = r ∗ cos(theta), y = r ∗ sin(theta)
        # Both sine and cosine of 45 degrees equal sqrt(2) / 2.
        circle_corner_distance = np.sqrt(2.0) / 2 * self.CENTER_CIRCLE_RADIUS

        self.keypoints["center_circle_top_left_corner"] = np.array(
            [-circle_corner_distance, -circle_corner_distance, 0],
            dtype=np.float32,
        )

        self.keypoints["center_circle_top_right_corner"] = np.array(
            [circle_corner_distance, -circle_corner_distance, 0],
            dtype=np.float32,
        )

        self.keypoints["center_circle_bottom_left_corner"] = np.array(
            [-circle_corner_distance, circle_corner_distance, 0],
            dtype=np.float32,
        )

        self.keypoints["center_circle_bottom_right_corner"] = np.array(
            [circle_corner_distance, circle_corner_distance, 0],
            dtype=np.float32,
        )

        self.keypoints["center_circle_left"] = np.array(
            [-self.CENTER_CIRCLE_RADIUS, 0, 0],
            dtype=np.float32,
        )

        self.keypoints["center_circle_right"] = np.array(
            [self.CENTER_CIRCLE_RADIUS, 0, 0],
            dtype=np.float32,
        )

        self.keypoints["left_penalty_ark_mark"] = np.array(
            [-self.PITCH_LENGTH / 2 + self.PENALTY_AREA_LENGTH, 0, 0],
            dtype=np.float32,
        )

        self.keypoints["left_penalty_ark_right_mark"] = (
            self.keypoints["left_penalty_mark"] + self.keypoints["center_circle_right"]
        )

        self.keypoints["right_penalty_ark_mark"] = np.array(
            [self.PITCH_LENGTH / 2 - self.PENALTY_AREA_LENGTH, 0, 0],
            dtype=np.float32,
        )

        self.keypoints["right_penalty_ark_left_mark"] = (
            self.keypoints["right_penalty_mark"] + self.keypoints["center_circle_left"]
        )

        intersections = find_circle_tangent_intersections(
            self.keypoints["center_mark"][:2],
            self.CENTER_CIRCLE_RADIUS,
            self.keypoints["halfway_top_touch_line_mark"][:2],
        )

        self.keypoints["center_circle_top_left_tangent"] = intersections[0]
        self.keypoints["center_circle_top_right_tangent"] = intersections[1]

        intersections = find_circle_tangent_intersections(
            self.keypoints["center_mark"][:2],
            self.CENTER_CIRCLE_RADIUS,
            self.keypoints["halfway_bottom_touch_line_mark"][:2],
        )

        self.keypoints["center_circle_bottom_left_tangent"] = intersections[0]
        self.keypoints["center_circle_bottom_right_tangent"] = intersections[1]

        self.keypoints["left_penalty_ark_top_tangent"] = (
            find_circle_tangent_intersections(
                self.keypoints["left_penalty_mark"][:2],
                self.CENTER_CIRCLE_RADIUS,
                self.keypoints["left_penalty_area_top_right_corner"][:2],
            )[1]
        )

        self.keypoints["left_penalty_ark_bottom_tangent"] = (
            find_circle_tangent_intersections(
                self.keypoints["left_penalty_mark"][:2],
                self.CENTER_CIRCLE_RADIUS,
                self.keypoints["left_penalty_area_bottom_right_corner"][:2],
            )[0]
        )

        self.keypoints["right_penalty_ark_top_tangent"] = (
            find_circle_tangent_intersections(
                self.keypoints["right_penalty_mark"][:2],
                self.CENTER_CIRCLE_RADIUS,
                self.keypoints["right_penalty_area_top_left_corner"][:2],
            )[0]
        )

        self.keypoints["right_penalty_ark_bottom_tangent"] = (
            find_circle_tangent_intersections(
                self.keypoints["right_penalty_mark"][:2],
                self.CENTER_CIRCLE_RADIUS,
                self.keypoints["right_penalty_area_bottom_left_corner"][:2],
            )[1]
        )

    def plot(self):
        keypoints = np.array(list(self.keypoints.values()))

        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=keypoints[:, 0],
                    y=keypoints[:, 1],
                    z=keypoints[:, 2],
                    mode="markers",
                    marker=dict(size=4),
                    text=list(self.keypoints.keys()),
                )
            ]
        )

        fig.update_scenes(
            aspectratio=dict(
                x=self.PITCH_LENGTH / 100,
                y=self.PITCH_WIDTH / 100,
                z=0.1,
            ),
            camera=dict(eye=dict(x=0, y=0.75, z=0.75), center=dict(x=0, y=0.15, z=0)),
            xaxis_autorange="reversed",
            zaxis_autorange="reversed",
        )

        fig.update_layout(
            autosize=True,
            dragmode=False,
            margin=dict(l=0, r=0, t=0, b=0),
        )

        return fig


KEYPOINTS_INDEX: SoccerPitchKeypoints[int] = {
    "center_mark": 0,
    "halfway_top_touch_line_mark": 1,
    "halfway_bottom_touch_line_mark": 2,
    "halfway_line_center_circle_top_mark": 3,
    "halfway_line_center_circle_bottom_mark": 4,
    "top_left_corner": 5,
    "top_right_corner": 6,
    "bottom_left_corner": 7,
    "bottom_right_corner": 8,
    "left_goal_top_left_post": 9,
    "left_goal_top_right_post": 10,
    "left_goal_bottom_left_post": 11,
    "left_goal_bottom_right_post": 12,
    "right_goal_top_left_post": 13,
    "right_goal_top_right_post": 14,
    "right_goal_bottom_left_post": 15,
    "right_goal_bottom_right_post": 16,
    "left_goal_area_top_left_corner": 17,
    "left_goal_area_top_right_corner": 18,
    "left_goal_area_bottom_left_corner": 19,
    "left_goal_area_bottom_right_corner": 20,
    "right_goal_area_top_left_corner": 21,
    "right_goal_area_top_right_corner": 22,
    "right_goal_area_bottom_left_corner": 23,
    "right_goal_area_bottom_right_corner": 24,
    "left_penalty_mark": 25,
    "right_penalty_mark": 26,
    "left_penalty_area_top_left_corner": 27,
    "left_penalty_area_top_right_corner": 28,
    "left_penalty_area_bottom_left_corner": 29,
    "left_penalty_area_bottom_right_corner": 30,
    "right_penalty_area_top_left_corner": 31,
    "right_penalty_area_top_right_corner": 32,
    "right_penalty_area_bottom_left_corner": 33,
    "right_penalty_area_bottom_right_corner": 34,
    "top_left_penalty_ark_mark": 35,
    "top_right_penalty_ark_mark": 36,
    "bottom_left_penalty_ark_mark": 37,
    "bottom_right_penalty_ark_mark": 38,
    "center_circle_top_left_corner": 39,
    "center_circle_top_right_corner": 40,
    "center_circle_bottom_left_corner": 41,
    "center_circle_bottom_right_corner": 42,
    "center_circle_left": 43,
    "center_circle_right": 44,
    "left_penalty_ark_mark": 45,
    "left_penalty_ark_right_mark": 46,
    "right_penalty_ark_mark": 47,
    "right_penalty_ark_left_mark": 48,
    "center_circle_top_left_tangent": 49,
    "center_circle_top_right_tangent": 50,
    "center_circle_bottom_left_tangent": 51,
    "center_circle_bottom_right_tangent": 52,
    "left_penalty_ark_top_tangent": 53,
    "left_penalty_ark_bottom_tangent": 54,
    "right_penalty_ark_top_tangent": 55,
    "right_penalty_ark_bottom_tangent": 56,
}

LINE_INTERSECTIONS = {
    0: ("Goal left crossbar", "Goal left post left "),
    1: ("Goal left crossbar", "Goal left post right"),
    2: ("Side line left", "Goal left post left "),
    3: ("Side line left", "Goal left post right"),
    4: ("Small rect. left main", "Small rect. left bottom"),
    5: ("Small rect. left main", "Small rect. left top"),
    6: ("Side line left", "Small rect. left bottom"),
    7: ("Side line left", "Small rect. left top"),
    8: ("Big rect. left main", "Big rect. left bottom"),
    9: ("Big rect. left main", "Big rect. left top"),
    10: ("Side line left", "Big rect. left bottom"),
    11: ("Side line left", "Big rect. left top"),
    12: ("Side line left", "Side line bottom"),
    13: ("Side line left", "Side line top"),
    14: ("Middle line", "Side line bottom"),
    15: ("Middle line", "Side line top"),
    16: ("Big rect. right main", "Big rect. right bottom"),
    17: ("Big rect. right main", "Big rect. right top"),
    18: ("Side line right", "Big rect. right bottom"),
    19: ("Side line right", "Big rect. right top"),
    20: ("Small rect. right main", "Small rect. right bottom"),
    21: ("Small rect. right main", "Small rect. right top"),
    22: ("Side line right", "Small rect. right bottom"),
    23: ("Side line right", "Small rect. right top"),
    24: ("Goal right crossbar", "Goal right post left"),
    25: ("Goal right crossbar", "Goal right post right"),
    26: ("Side line right", "Goal right post left"),
    27: ("Side line right", "Goal right post right"),
    28: ("Side line right", "Side line bottom"),
    29: ("Side line right", "Side line top"),
}


def find_closest_point(points: np.ndarray, point: np.ndarray):
    distances = np.linalg.norm(points - point, axis=1)
    closest_index = np.argmin(distances)

    return points[closest_index]


def find_intersection(line1: np.ndarray, line2: np.ndarray):
    while True:
        x1, y1 = line1[:, 0], line1[:, 1]
        x2, y2 = line2[:, 0], line2[:, 1]

        x1_mean, x2_mean = np.mean(x1), np.mean(x2)

        x1_vertical = np.all(np.isclose(x1, x1_mean, atol=0.5))
        x2_vertical = np.all(np.isclose(x2, x2_mean, atol=0.5))

        if x1_vertical:
            # Two lines must not be vertical at the same time.
            if x2_vertical:
                return None

            m2, c2 = P.polyfit(x2, y2, deg=1)
            x = x1_mean
            y = m2 * x + c2
        elif x2_vertical:
            m1, c1 = P.polyfit(x1, y1, deg=1)
            x = x2_mean
            y = m1 * x + c1
        else:
            m1, c1 = P.polyfit(x1, y1, deg=1)
            m2, c2 = P.polyfit(x2, y2, deg=1)
            x = (c2 - c1) / (m1 - m2) if m1 - m2 != 0 else 0
            y = m1 * x + c1

        if not line1.shape[0] > 2 and not line2.shape[0] > 2:
            return x, y

        line1 = find_closest_point(line1, np.array((x, y)))
        line2 = find_closest_point(line2, np.array((x, y)))


def find_ellipse_tangent_intersections(coefficients: tuple, point: np.ndarray):
    a, b, c, d, e, f = coefficients
    x1, y1 = point

    A = (
        4 * a * c * x1 * y1
        + 2 * a * e * x1
        - b**2 * x1 * y1
        - b * d * x1
        - b * e * y1
        - 2 * b * f
        + 2 * c * d * y1
        + d * e
    )

    B = 2 * np.sqrt(
        (-4 * a * c * f + a * e**2 + b**2 * f - b * d * e + c * d**2)
        * (a * x1**2 + b * x1 * y1 + c * y1**2 + d * x1 + e * y1 + f)
    )

    C = (
        4 * a * c * x1**2
        - b**2 * x1**2
        - 2 * b * e * x1
        + 4 * c * d * x1
        + 4 * c * f
        - e**2
    )

    gradients = (A - B) / C, (A + B) / C
    points = []

    for m in gradients:
        D = b * m * x1 - b * y1 + 2 * c * m**2 * x1 - 2 * c * m * y1 - d - e * m
        E = 2 * (a + b * m + c * m**2)

        x = D / E
        y = y1 + m * x - m * x1

        points.append((x, y))

    return points


def find_ellipse_line_intersection(coefficients: tuple):
    if len(coefficients) < 6:
        return None

    #
    pass


def get_intersection_points():
    pass


def get_ellipse_points(
    points: dict[str, np.ndarray],
    pitch_keypoints: np.ndarray[tuple[Literal[57], Literal[2]]],
):
    circles = ["Circle central", "Circle left", "Circle right"]

    circle_points = {
        "Circle central": {
            "halfway_line_center_circle_top_mark": {
                "type": "intersection",
            },
            "halfway_line_center_circle_bottom_mark": {
                "type": "intersection",
            },
        },
        "Circle left": {
            "top_left_penalty_ark_mark": {
                "type": "intersection",
            },
            "bottom_left_penalty_ark_mark": {
                "type": "intersection",
            },
        },
    }

    for circle in circles:
        if circle not in points:
            continue

        # A minimum of five points is required to define a unique ellipse.
        # https://sarcasticresonance.wordpress.com/2012/05/14/how-many-points-does-it-take-to-define/
        if len(points[circle] < 5):
            continue

        reg = LsqEllipse().fit(points[circle])
        ellipse = reg.coefficients if len(reg.coefficients) == 6 else None

        if ellipse is None:
            continue

        if circle == "Circle central":
            idx1 = KEYPOINTS_INDEX["halfway_top_touch_line_mark"]
            idx2 = KEYPOINTS_INDEX["center_circle_top_left_tangent"]
            idx3 = KEYPOINTS_INDEX["center_circle_top_right_tangent"]
            idx4 = KEYPOINTS_INDEX["halfway_bottom_touch_line_mark"]
            idx5 = KEYPOINTS_INDEX["center_circle_bottom_left_tangent"]
            idx6 = KEYPOINTS_INDEX["center_circle_bottom_right_tangent"]
            idx7 = KEYPOINTS_INDEX["halfway_line_center_circle_top_mark"]
            idx8 = KEYPOINTS_INDEX["halfway_line_center_circle_bottom_mark"]

            points_left = find_ellipse_tangent_intersections(
                reg.coefficients,
                pitch_keypoints[idx1],
            )

            pitch_keypoints[idx2] = points_left[0]
            pitch_keypoints[idx3] = points_left[1]

            points_right = find_ellipse_tangent_intersections(
                reg.coefficients,
                pitch_keypoints[idx4],
            )

            pitch_keypoints[idx5] = points_right[0]
            pitch_keypoints[idx6] = points_right[1]

            # find intersection of the ellipse with the line
            # points["Middle line"]
        elif circle == "Circle left":
            idx1 = KEYPOINTS_INDEX["left_penalty_area_top_right_corner"]
            idx2 = KEYPOINTS_INDEX["left_penalty_ark_top_tangent"]
            idx3 = KEYPOINTS_INDEX["left_penalty_area_bottom_right_corner"]
            idx4 = KEYPOINTS_INDEX["left_penalty_ark_bottom_tangent"]

            points_top = find_ellipse_tangent_intersections(
                reg.coefficients,
                pitch_keypoints[idx1],
            )

            pitch_keypoints[idx2] = points_top[1]

            points_bottom = find_ellipse_tangent_intersections(
                reg.coefficients,
                pitch_keypoints[idx3],
            )

            pitch_keypoints[idx4] = points_bottom[1]
        elif circle == "Circle right":
            idx1 = KEYPOINTS_INDEX["right_penalty_area_top_left_corner"]
            idx2 = KEYPOINTS_INDEX["right_penalty_ark_top_tangent"]
            idx3 = KEYPOINTS_INDEX["right_penalty_area_bottom_left_corner"]
            idx4 = KEYPOINTS_INDEX["right_penalty_ark_bottom_tangent"]

            points_top = find_ellipse_tangent_intersections(
                reg.coefficients,
                pitch_keypoints[idx1],
            )

            pitch_keypoints[idx2] = points_top[0]

            points_bottom = find_ellipse_tangent_intersections(
                reg.coefficients,
                pitch_keypoints[idx3],
            )

            pitch_keypoints[idx4] = points_bottom[0]


def get_pitch_points():
    # Get the intersection points of the lines.
    # Get the ellipse points (TODO: get intersections between circles and lines)
    # Get homography points
    pass


class Test:
    @staticmethod
    def test_find_closest_point():
        points = np.array([(0, 0), (1, 1), (2, 2)])
        point = np.array((0, 1))

        closest_point = find_closest_point(points, point)

        assert np.all(closest_point == np.array((0, 0)))
