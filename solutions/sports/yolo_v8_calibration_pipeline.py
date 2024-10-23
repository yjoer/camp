import numpy as np
import plotly.graph_objects as go
import torch
import torchvision.transforms.v2.functional as tvf


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

        self.keypoints = {
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
