# %%
import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
from typing import cast

import fsspec
from minio import Minio
from s3fs import S3FileSystem

# %%
DATASET_PATH = "s3://datasets/soccernet_calibration_2023"

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

annotations_np = fs.glob(f"{subset_path}/*.json")
annotations_np = sorted(annotations_np)
annotations = [fs.unstrip_protocol(a) for a in annotations_np]

annotations_np = annotations_np[:5000]
annotations = annotations[:5000]

# %% [markdown]
# ## Single-Threaded

# %%
# %%timeit -n1 -r3

with fsspec.open_files(annotations, "r", **storage_options) as files:
    for f in files:
        json.load(f)

# %% [markdown]
# ## Multithreaded

# %%
# %%timeit -n1 -r3

pool = ThreadPoolExecutor()


def fsspec_read_json(f):
    return json.load(f)


with fsspec.open_files(annotations, "r", **storage_options) as files:
    results_fsspec = [x for x in pool.map(fsspec_read_json, files)]

# %% [markdown]
# ## Multithreaded MinIO

# %%
minio = Minio(
    endpoint=cast(str, storage_options["endpoint_url"])
    .replace("http://", "")
    .replace("https://", ""),
    access_key=storage_options["key"],
    secret_key=storage_options["secret"],
    secure=False,
)

# %%
# %%timeit -n1 -r3

pool = ThreadPoolExecutor()


def minio_read_json(f):
    segments = f.split("/")
    bucket_name = segments[0]
    object_name = "/".join(segments[1:])

    response = minio.get_object(bucket_name, object_name)
    data = response.read()
    response.release_conn()

    return json.loads(data)


results_minio = [x for x in pool.map(minio_read_json, annotations_np)]

# %% [markdown]
# ## Async IO

# %%
# %%timeit -n1 -r3


def target_fn():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    fs = fsspec.filesystem(
        protocol,
        **storage_options,
        asynchronous=True,
        loop=loop,
        skip_instance_cache=True,
    )

    async def gather():
        async def read_json(f: str):
            f = await fs.open_async(f, "rb")
            data = await f.read()
            await f.close()

            return json.loads(data)

        return await asyncio.gather(*[read_json(f) for f in annotations_np])

    loop.run_until_complete(gather())
    S3FileSystem.close_session(loop, fs)


t = Thread(target=target_fn)
t.start()
t.join()

# %%
