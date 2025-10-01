import io
import json
from typing import Any

import fsspec
from PIL import Image


class VitroxBody1k:
    def __init__(self) -> None:
        pass

    @staticmethod
    def load(
        path: str,
        storage_options: dict | None = None,
    ) -> dict[str, list[Any] | list[list[Any]]]:
        if storage_options is None:
            storage_options = {}

        arrays: dict[str, list[Any] | list[list[Any]]] = {}

        for subset in ["train", "test"]:
            with fsspec.open(f"{path}/{subset}.json", **storage_options) as f:
                annotation = json.load(f)
                files = [f"{path}/{i['file_name']}" for i in annotation["images"]]
                labels = [[a["category_id"] + 1] for a in annotation["annotations"]]
                boxes = [[a["bbox"][:4]] for a in annotation["annotations"]]
                thetas = [[a["bbox"][4]] for a in annotation["annotations"]]

            with fsspec.open_files(files, **storage_options) as files:
                load_image = lambda x: Image.open(io.BytesIO(x.read())).convert("RGB")
                images = [load_image(f) for f in files]

            arrays[subset] = images
            arrays[f"{subset}_labels"] = labels
            arrays[f"{subset}_boxes"] = boxes
            arrays[f"{subset}_thetas"] = thetas

        return arrays
