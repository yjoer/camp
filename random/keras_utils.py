# %%
import os

os.environ["KERAS_BACKEND"] = "torch"

# %%
import keras

# %%
pbar = keras.utils.Progbar(target=10)

for i in range(10):
    pbar.update(current=i + 1, values=[("loss", i + 1)])

pbar._values

# %%
