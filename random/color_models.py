# %%
import numpy as np


# %%
def rgb_to_cmy(pixel: list | tuple):
    return 1 - np.array(pixel) / 255


# %%
rgb_to_cmy((100, 100, 100))


# %%
def rgb_to_hsi(pixel: list | tuple, normalize: bool = False):
    if normalize:
        pixel = tuple(x / 255 for x in pixel)

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
hsi = rgb_to_hsi((50, 10, 200), normalize=True)
hsi


# %%
def hsi_to_rgb(pixel: list | tuple):
    H, S, I = pixel
    cosd = lambda x: np.cos(np.deg2rad(x))

    if H >= 0 and H < 120:
        R = I * (1 + S * cosd(H) / cosd(60 - H))
        B = I * (1 - S)
        G = 3 * I - (R + B)

    if H >= 120 and H < 240:
        R = I * (1 - S)
        G = I * (1 + S * cosd(H - 120) / cosd(60 - (H - 120)))
        B = 3 * I - (R + G)

    if H >= 240 and H < 360:
        G = I * (1 - S)
        B = I * (1 + S * cosd(H - 240) / cosd(60 - (H - 240)))
        R = 3 * I - (G + B)

    return tuple(round(x * 255) for x in (R, G, B))


# %%
hsi_to_rgb(hsi)

# %%
