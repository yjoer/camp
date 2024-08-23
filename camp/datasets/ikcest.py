import fsspec
import numpy as np
import torch
import torchvision.transforms.v2.functional as tvf
from PIL import Image
from torch.utils.data import Dataset


class IKCEST:
    def __init__(self):
        pass

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

    def __init__(self, path: str, subset: str, storage_options={}):
        frames, annotations_dict = IKCEST.load(path, subset, storage_options)

        self.path = path
        self.subset = subset
        self.storage_options = storage_options
        self.frames = frames
        self.annotations_dict = annotations_dict

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx: int):
        frame_path = self.frames[idx]
        video_name = frame_path.split("/")[-3]

        with fsspec.open(frame_path, **self.storage_options) as f:
            frame = Image.open(f).convert("RGB")
            frame = tvf.to_image(frame)
            frame = tvf.to_dtype(frame, dtype=torch.float32, scale=True)

        annotations = self.annotations_dict[video_name]
        annotation = annotations[annotations[:, 0] == idx + 1]
        boxes = torch.from_numpy(annotation[:, 2:6])

        labels = torch.ones((len(boxes)))

        return frame, {"labels": labels, "boxes": boxes}
