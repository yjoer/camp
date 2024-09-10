from configparser import ConfigParser
from typing import Callable
from typing import Optional

import fsspec
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class IKCEST:
    def __init__(self):
        pass

    @staticmethod
    def load_test(path: str, storage_options={}):
        protocol = fsspec.utils.get_protocol(path)
        subset_path = f"{path}/test"

        fs = fsspec.filesystem(protocol, **storage_options)
        videos = fs.ls(subset_path)
        videos = [fs.unstrip_protocol(v) for v in videos]
        videos = sorted(videos)

        subset_frames = []
        subset_seq_metadata = []

        for video in videos:
            frames = fs.ls(f"{video}/img1")
            frames = [fs.unstrip_protocol(f) for f in frames]
            frames = sorted(frames)
            subset_frames.extend(frames)

            with fsspec.open(f"{video}/seqinfo.ini", "r", **storage_options) as f:
                config = ConfigParser()
                config.read_file(f)

                seq_metadata = {
                    "name": config["Sequence"]["name"],
                    "frame_rate": config["Sequence"]["frameRate"],
                    "sequence_length": config["Sequence"]["seqLength"],
                    "image_width": config["Sequence"]["imWidth"],
                    "image_height": config["Sequence"]["imHeight"],
                    "image_extension": config["Sequence"]["imExt"],
                }

                subset_seq_metadata.append(seq_metadata)

        return subset_frames, subset_seq_metadata

    @staticmethod
    def load(path: str, subset: str, storage_options={}):
        """
        Retrieve the list of images and load all associated annotations into memory,
        considering their small size. Organize these annotations in a dictionary where
        each key represents a video name and the corresponding value is a NumPy array
        containing the set of annotations related to that video.
        """
        protocol = fsspec.utils.get_protocol(path)

        fs = fsspec.filesystem(protocol, **storage_options)
        videos = fs.ls(f"{path}/{subset}")
        frames = fs.glob(f"{path}/{subset}/**/img1/*.jpg")
        frames = [fs.unstrip_protocol(f) for f in frames]

        annotations = [fs.unstrip_protocol(f"{v}/gt/gt.txt") for v in videos]
        annotations_dict = {}

        with fsspec.open_files(annotations, **storage_options) as files:
            for video, f in zip(videos, files):
                video_name = video.split("/")[-1]
                annotations_dict[video_name] = np.loadtxt(f, delimiter=",")

        return frames, annotations_dict


class IKCESTDetectionDataset(Dataset):
    """
    Given an index, load the frame, and filter the corresponding annotations to extract
    the bounding boxes. The tracklet identifier is disregarded since it is irrelevant to
    the player detection task.
    """

    def __init__(
        self,
        path: str,
        subset: str,
        storage_options={},
        transforms: Optional[Callable] = None,
    ):
        frames, annotations_dict = IKCEST.load(path, subset, storage_options)

        self.path = path
        self.subset = subset
        self.storage_options = storage_options
        self.frames = frames
        self.annotations_dict = annotations_dict
        self.transforms = transforms

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx: int):
        frame_path = self.frames[idx]
        video_name = frame_path.split("/")[-3]

        with fsspec.open(frame_path, **self.storage_options) as f:
            frame = Image.open(f).convert("RGB")

        annotations = self.annotations_dict[video_name]
        annotation = annotations[annotations[:, 0] == (idx % 750) + 1]
        boxes = torch.from_numpy(annotation[:, 2:6])

        labels = torch.zeros((len(boxes)))

        if self.transforms is not None:
            target = {"labels": labels, "boxes": boxes}
            frame, target = self.transforms(frame, target)

        return frame, target


class IKCESTDetectionTestDataset(Dataset):
    def __init__(
        self,
        path: str,
        storage_options={},
        transforms: Optional[Callable] = None,
    ):
        frames, seq_metadata = IKCEST.load_test(path, storage_options)

        self.frames = frames
        self.seq_metadata = seq_metadata
        self.storage_options = storage_options
        self.transforms = transforms

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx: int):
        frame_path = self.frames[idx]

        with fsspec.open(frame_path, **self.storage_options) as f:
            frame = Image.open(f).convert("RGB")

        if self.transforms is not None:
            frame = self.transforms(frame)

        metadata = {}
        metadata["seq"] = self.seq_metadata[idx // 750]

        return frame, metadata
