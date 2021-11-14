import pandas as pd
from dataclasses import dataclass
from typing import List
import os
from torch.utils.data import Dataset
import numpy as np
import pickle
from dataloaders.rawvideo_util import RawVideoExtractor


base = '.'


@dataclass(frozen=True)
class Video:
    package_id: int
    start: int
    end: int

    @property
    def path(self):
        return f'{base}/clips/{self.package_id}-{self.start}-{self.end}.mp4'

    @classmethod
    def from_row(cls, first: bool, row: pd.Series) -> "Video":
        pre = f"clip{1 if first else 2}"
        return cls(row[f"{pre}_package_id"], row[f"{pre}_start"], row[f"{pre}_end"])


@dataclass
class VideoPair:
    v1: Video
    v2: Video


def ds_from_path(path_csv: str) -> List[VideoPair]:
    df = pd.read_csv(path_csv)
    return [
        VideoPair(Video.from_row(first=True, row=row), Video.from_row(first=False, row=row))
        for _, row in df.iterrows()
    ]


ds = dict(
    train=ds_from_path(f"{base}/train.csv"),
    validation=ds_from_path(f"{base}/validation.csv")
)


class MatchCutFrameDataLoader(Dataset):
    def __init__(
        self,
        partition: str,
        frame_rate: int,
        image_size: int,
        max_frames: int,
        slice_framepos: int,  # TODO: what is this?
        frame_order: int,  # TODO: what is this?
    ):
        self.partition = partition
        self.frame_rate = frame_rate
        self.image_size = image_size
        self.max_frames = max_frames
        self.slice_framepos = slice_framepos
        self.frame_order = frame_order

        self.pairs = ds[partition]
        self.video_extractor = RawVideoExtractor(framerate=frame_rate, size=image_size)

    def __len__(self):
        return len(self.pairs)

    @property
    def vid_size(self) -> int:
        return self.video_extractor.size

    def _get_video(self, video_path: str):
        print('video path is', video_path)
        vid_data = self.video_extractor.get_video_data(video_path)['video']
        vid_slice = self.video_extractor.process_raw_data(vid_data)
        if self.max_frames < vid_slice.shape[0]:
            if self.slice_framepos == 0:
                vid_slice = vid_slice[:self.max_frames, ...]
            elif self.slice_framepos == 1:
                vid_slice = vid_slice[-self.max_frames:, ...]
            else:
                sample_idx = np.linspace(0, vid_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                vid_slice = vid_slice[sample_idx, ...]

        vid_slice = self.video_extractor.process_frame_order(vid_slice, frame_order=self.frame_order)
        slice_len = len(vid_slice)
        video = np.zeros((1, self.max_frames, 1, 3, self.vid_size, self.vid_size))
        mask = np.zeros((1, self.max_frames), dtype=np.long)
        video[0][:slice_len, ...] = vid_slice
        mask[0][:slice_len] = 1
        return video, mask

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        v1, m1 = self._get_video(pair.v1.path)
        v2, m2 = self._get_video(pair.v2.path)
        return v1, m1, v2, m2
