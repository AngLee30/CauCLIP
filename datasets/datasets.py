import os
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from PIL import Image

from .augmentation import get_augmentation, colorful_spectrum_mix

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self): # 0, 1, 2
        return int(self._data[2])
    
    @property
    def background_type(self): 
        # 0: clinical surgical background
        # 1: pumpkin background (simulated)                    
        return int(self._data[3])


class SurgVisDom(data.Dataset):
    def __init__(
            self, 
            list_file, 
            labels_file,
            root_dir,
            seg_num=1, # number of segments in each video
            seg_length=1, # number of frames in each segment            
            image_tmpl=None, 
            transform=None,
            random_shift=True, 
            index_bias=1,
            training_mode=True,
            alpha=0.5
        ):

        self.list_file = list_file
        self.labels_file = labels_file
        self.root_dir = root_dir
        self.seg_num = seg_num
        self.seg_length = seg_length
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.index_bias = index_bias
        self.training_mode = training_mode
        self.alpha = alpha
        self.loop = False

        self._parse_list()

    def _parse_list(self):
        rows = [row.strip().split(' ') for row in open(self.list_file)] # list[list[str]]
        self.video_list: list[VideoRecord] = []
        self.background_group = {0: [], 1: []}

        for idx, row in enumerate(rows):
            base: str = row[0]
            path = os.path.join(self.root_dir, base)
            num_frames: str = row[1]
            label: str = row[2]

            if self.training_mode:
                background_type: str = row[3]
                self.video_list.append(VideoRecord([path, num_frames, label, background_type]))
                self.background_group[int(background_type)].append(idx) # idx -> video_list[idx] -> VideoRecord
            else:
                self.video_list.append(VideoRecord([path, num_frames, label, -1]))

    @property
    def total_length(self):
        return self.seg_num * self.seg_length # total number of frames to sample from a video
    
    @property
    def classes(self):
        """
        [[0, '...'], [1, '...'], [2, '...']]
        """
        classes_all = pd.read_csv(self.labels_file)
        return classes_all.values.tolist()
    
    def __len__(self):
        return len(self.video_list)

    def _sample_indices(self, record: VideoRecord) -> np.ndarray:
        # record: path, num_frames, label
        if record.num_frames <= self.total_length:
            if self.loop:
                start_offset = np.random.randint(1, record.num_frames) # [1, record.num_frames)
                return np.mod(np.arange(self.total_length) + start_offset, record.num_frames) + self.index_bias
            else:
                offsets = np.concatenate((
                    np.arange(1, record.num_frames), # [1, record.num_frames)
                    np.random.randint(1, record.num_frames, size=self.total_length - record.num_frames + 1)
                ))
                return np.sort(offsets)[:self.total_length] + self.index_bias
        else: # record.num_frames > self.total_length
            offsets = list()
            interval = record.num_frames // self.seg_num
            ticks = [i * interval for i in range(self.seg_num + 1)] # [0, self.seg_num]

            for i in range(self.seg_num): # [0, self.seg_num)
                tick_len = ticks[i + 1] - ticks[i]
                tick = ticks[i]
                if tick_len >= self.seg_length:
                    tick += np.random.randint(0, tick_len - self.seg_length + 1)
                offsets.extend([j for j in range(tick, tick + self.seg_length)])
            return np.array(offsets) + self.index_bias

    def _get_val_indices(self, record: VideoRecord) -> np.ndarray:
        if self.seg_num == 1:
            return np.array([record.num_frames // 2], dtype=int) + self.index_bias
        
        if record.num_frames <= self.total_length:
            if self.loop:
                return np.mod(np.arange(self.total_length), record.num_frames) + self.index_bias
            return np.array([i * record.num_frames // self.total_length
                             for i in range(self.total_length)], dtype=int) + self.index_bias

        offset = (record.num_frames / self.seg_num - self.seg_length) / 2.0
        ret =  np.array(
            [
                i * record.num_frames / self.seg_num + offset + j
                    for i in range(self.seg_num)
                    for j in range(self.seg_length)
            ], 
            dtype=int
        ) + self.index_bias
        return ret

    def __getitem__(self, index):
        return self.get(index)

    def get(self, index: int):
        curr_record: VideoRecord = self.video_list[index]
        curr_record_indices = self._sample_indices(curr_record) if self.random_shift else self._get_val_indices(curr_record)
        curr_images = list()
        for idx in curr_record_indices:
            try:
                seg_imgs = self._load_image(curr_record.path, idx)
            except OSError:
                print(f"ERROR: Could not read frame {self.image_tmpl.format(idx)} from {curr_record.path}")
                raise
            curr_images.extend(seg_imgs) # list: [PIL.Image.Image]

        original_data = self.transform(curr_images)

        if not self.training_mode: # not use background augmentation
            return original_data, curr_record.label # (T*3, 224, 224)

        else: # use background augmentation
            other_index = np.random.choice(self.background_group[1 - curr_record.background_type])
            other_record: VideoRecord = self.video_list[other_index]
            other_record_indices = self._sample_indices(other_record) if self.random_shift else self._get_val_indices(other_record)
            other_images = list()
            for idx in other_record_indices:
                try:
                    seg_imgs = self._load_image(other_record.path, idx)
                except OSError:
                    print(f"ERROR: Could not read frame {self.image_tmpl.format(idx)} from {other_record.path}")
                    raise
                other_images.extend(seg_imgs) # list: [PIL.Image.Image]
            
            aug_images = list()
            for curr_img, other_img in zip(curr_images, other_images):
                aug_img = colorful_spectrum_mix(np.array(curr_img), np.array(other_img), self.alpha) # ndarray
                aug_img = Image.fromarray(aug_img) # ndarray -> PIL.Image
                aug_images.append(aug_img)

            augmented_data = self.transform(aug_images)
            return original_data, augmented_data, curr_record.label # (T*3, 224, 224)

    def _load_image(self, directory, idx):
        return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')] # [PIL.Image.Image]


def visualize_sample(sample: torch.Tensor, T: int, save: bool = False):
    # sample: [1, T*3, 224, 224]
    assert sample.shape[0] == 1 and sample.shape[1] >= 3
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3,1,1)
    std  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3,1,1)

    rows = (T + 7) // 8
    cols = 8
    _, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    if rows == 1:
        axes = np.array([axes])
    sample = sample.view((-1, T, 3) + sample.size()[-2:]) # [1, T*3, 224, 224] -> [1, T, 3, 224, 224]

    for i in range(T):
        r = i // 8
        c = i % 8       
        img = (sample[0, i].cpu()) * std + mean
        img = img.clamp(0, 1).permute(1, 2, 0).numpy() # (C, H, W) -> (H, W, C)       
        axes[r, c].imshow(img)
        axes[r, c].axis("off")

    plt.tight_layout()
    if save:
        plt.savefig("./visualize_sample.png")
    else:
        plt.show()