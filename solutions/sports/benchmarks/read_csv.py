# %%
import io
import os
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

import fsspec
import numpy as np
import pandas as pd
from pyarrow import csv

# %%
DATASET_PATH = "s3://datasets/ikcest_2024"

# %%
storage_options = {
    "endpoint_url": os.getenv("S3_ENDPOINT"),
    "key": os.getenv("S3_ACCESS_KEY_ID"),
    "secret": os.getenv("S3_SECRET_ACCESS_KEY"),
}

if not os.getenv("S3_ENDPOINT"):
    storage_options = {}

# %%
subset_path = f"{DATASET_PATH}/train"
protocol = fsspec.utils.get_protocol(subset_path)
fs = fsspec.filesystem(protocol, **storage_options)

videos = fs.ls(subset_path)
videos = [fs.unstrip_protocol(v) for v in videos]
videos = sorted(videos)

annotations = [f"{v}/gt/gt.txt" for v in videos]

# %% [markdown]
# ## Parsing

# %%
with fsspec.open(f"{subset_path}/SNMOT-060/gt/gt.txt", **storage_options) as f:
    x = io.BytesIO(f.read())

# %%
# %%timeit
x.seek(0)

# %%
# %%timeit
x.seek(0)
np_pd = pd.read_csv(x, header=None).to_numpy()

# %%
# %%timeit
x.seek(0)
read_options = csv.ReadOptions(autogenerate_column_names=True)
np_pa = csv.read_csv(x, read_options).to_pandas().to_numpy()

# %%
# %%timeit
x.seek(0)
np_lt = np.loadtxt(x, delimiter=",")

# %% [markdown]
# ## Single File

# %%
# %%timeit
with fsspec.open(f"{subset_path}/SNMOT-060/gt/gt.txt", **storage_options) as f:
    np_pd = pd.read_csv(f, header=None).to_numpy()

# %%
# %%timeit
read_options = csv.ReadOptions(autogenerate_column_names=True)

with fsspec.open(f"{subset_path}/SNMOT-060/gt/gt.txt", **storage_options) as f:
    np_pa = csv.read_csv(f, read_options).to_pandas().to_numpy()

# %%
# %%timeit
with fsspec.open(f"{subset_path}/SNMOT-060/gt/gt.txt", **storage_options) as f:
    np_lt = np.loadtxt(f, delimiter=",")

# %% [markdown]
# ## Multiple Files

# %%
# %%timeit
with fsspec.open_files(annotations, **storage_options) as files:
    for f in files:
        np_pd = pd.read_csv(f, header=None).to_numpy()

# %%
# %%timeit
read_options = csv.ReadOptions(autogenerate_column_names=True)

with fsspec.open_files(annotations, **storage_options) as files:
    for f in files:
        np_pa = csv.read_csv(f, read_options).to_pandas().to_numpy()

# %%
# %%timeit
with fsspec.open_files(annotations, **storage_options) as files:
    for f in files:
        np_lt = np.loadtxt(f, delimiter=",")

# %% [markdown]
# ## Multiple Files Multithreaded

# %%
# %%timeit
pool = ThreadPoolExecutor()


def pd_read_csv(f):
    return pd.read_csv(f, header=None).to_numpy()


with fsspec.open_files(annotations, **storage_options) as files:
    futures = []
    results_pd = []

    for f in files:
        futures.append(pool.submit(pd_read_csv, f))

    for future in as_completed(futures):
        results_pd.append(future.result())

# %%
# %%timeit
pool = ThreadPoolExecutor()
read_options = csv.ReadOptions(autogenerate_column_names=True)


def pa_read_csv(f):
    return csv.read_csv(f, read_options).to_pandas().to_numpy()


with fsspec.open_files(annotations, **storage_options) as files:
    futures = []
    results_pa = []

    for f in files:
        futures.append(pool.submit(pa_read_csv, f))

    for future in as_completed(futures):
        results_pa.append(future.result())

# %%
# %%timeit
pool = ThreadPoolExecutor()


def np_read_csv(f):
    return np.loadtxt(f, delimiter=",")


with fsspec.open_files(annotations, **storage_options) as files:
    futures = []
    results_np = []

    for f in files:
        futures.append(pool.submit(np_read_csv, f))

    for future in as_completed(futures):
        results_np.append(future.result())

# %%
