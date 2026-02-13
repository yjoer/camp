# %%
import numpy as np


# %%
def rgb_to_cmy(pixel: list | tuple, normalize: bool = False) -> tuple:
  pixel_np = np.array(pixel)

  if normalize:
    pixel_np = pixel_np / 255  # noqa: PLR6104

  return tuple(1 - pixel_np)


# %%
cmy = rgb_to_cmy((100, 100, 100), normalize=True)
cmy


# %%
def cmy_to_rgb(pixel: list | tuple) -> tuple:
  return tuple(round(x) for x in tuple((1 - np.array(pixel)) * 255))


# %%
cmy_to_rgb(cmy)


# %%
def cmy_to_cmyk(pixel: list | tuple) -> tuple:
  K = np.min(pixel)

  if K == 1:
    return 0, 0, 0, K

  C, M, Y = pixel

  C = (C - K) / (1 - K)
  M = (M - K) / (1 - K)
  Y = (Y - K) / (1 - K)

  return C, M, Y, K


# %%
cmyk = cmy_to_cmyk(cmy)
cmyk


# %%
def cmyk_to_cmy(pixel: list | tuple) -> tuple:
  C, M, Y, K = pixel

  C = C * (1 - K) + K
  M = M * (1 - K) + K
  Y = Y * (1 - K) + K

  return C, M, Y


# %%
cmyk_to_cmy(cmyk)


# %%
def rgb_to_hsi(pixel: list | tuple, normalize: bool = False) -> tuple:
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
def hsi_to_rgb(pixel: list | tuple) -> tuple:
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
