# %%
import numpy as np


# %%
def rgb_to_cmy(pixel: list | tuple):
    return 1 - np.array(pixel) / 255


# %%
rgb_to_cmy((100, 100, 100))


# %%
def rgb_to_hsi(pixel: list | tuple):
    R, G, B = pixel

    numerator = ((R - G) + (R - B)) / 2
    denominator = np.sqrt((R - G) ** 2 + (R - B) * (G - B))

    H = np.rad2deg(np.arccos(numerator / denominator))

    if B > G:
        H = 360 - H

    S = 1 - (3 / (R + G + B)) * np.min(pixel)
    I = (R + G + B) / 3

    return H, S, I


# %%
rgb_to_hsi([50 / 255, 10 / 255, 200 / 255])

# %%
