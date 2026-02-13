# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import NMF
from sklearn.decomposition import PCA

# %matplotlib inline
# %config InlineBackend.figure_formats = ['retina']

# %%
segments: dict[int, int] = {
  0: 0b1111110,
  1: 0b0110000,
  2: 0b1101101,
  3: 0b1111001,
  4: 0b0110011,
  5: 0b1011011,
  6: 0b1011111,
  7: 0b1110000,
  8: 0b1111111,
  9: 0b1111011,
}


def generate_led_digit(
  digit: int | None = None,
  segment: int | None = None,
  shape: tuple = (11, 6),
  pad: int = 1,
) -> np.ndarray:
  if digit is not None: segment = segments[digit]
  if segment is None: return np.zeros(shape)

  height, width = shape
  segment_height = (height - 3) // 2
  middle_row = segment_height + 1

  bitmap = np.zeros((height, width), dtype=int)

  if segment & 0b1000000:
    bitmap[0, 1 : width - 1] = 1
  if segment & 0b0100000:
    bitmap[1:middle_row, width - 1] = 1
  if segment & 0b0010000:
    bitmap[middle_row + 1 : height - 1, width - 1] = 1
  if segment & 0b0001000:
    bitmap[height - 1, 1 : width - 1] = 1
  if segment & 0b0000100:
    bitmap[middle_row + 1 : height - 1, 0] = 1
  if segment & 0b0000010:
    bitmap[1:middle_row, 0] = 1
  if segment & 0b0000001:
    bitmap[middle_row, 1 : width - 1] = 1

  return np.pad(bitmap, pad_width=pad, constant_values=0)


# %%
plt.figure(constrained_layout=True)

for i in range(10):
  plt.subplot(1, 10, i + 1)
  plt.imshow(generate_led_digit(digit=i), cmap="gray")
  plt.yticks([])
  plt.xticks([])

# %%
not_segment = lambda x: x == 0 or (x & (x - 1)) != 0
digits = [generate_led_digit(segment=i) for i in range(128) if not_segment(i)]
digits = np.array(digits).reshape((121, 104))

# %%
nmf = NMF(n_components=7, random_state=26)
nmf.fit(digits)

for i, component in enumerate(nmf.components_):
  plt.subplot(1, 8, i + 1)
  plt.imshow(component.reshape((13, 8)), cmap="gray")
  plt.yticks([])
  plt.xticks([])

# %%
pca = PCA(n_components=7, random_state=26)
pca.fit(digits)

for i, component in enumerate(pca.components_):
  plt.subplot(1, 8, i + 1)
  plt.imshow(component.reshape((13, 8)), cmap="gray")
  plt.yticks([])
  plt.xticks([])

# %%
