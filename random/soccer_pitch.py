# %%
from solutions.sports.yolo_v8_calibration_pipeline import SoccerPitch

# %load_ext autoreload
# %autoreload 2

# %%
pitch = SoccerPitch()

# %%
fig = pitch.plot()
fig.show()

# %%
