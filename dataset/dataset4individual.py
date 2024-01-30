import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from util.data import process_video


class Dataset4Individual(Dataset):
    def __init__(self, id, phase, seeg_dir, video_dir, video_processor_ckpt="MCG-NJU/videomae-base",
                 num_frame_2_sample=16):
        self.id = id
        self.phase = phase
        self.seeg_dir = seeg_dir
        self.video_dir = video_dir
        self.video_processor_ckpt = video_processor_ckpt
        self.num_frame_2_sample = num_frame_2_sample

        df = pd.read_csv(os.path.join(self.seeg_dir, f'{self.id}_preprocessed_data.csv'))
        df = df[df['Phase'] == self.phase]
        df = df.drop(['Error_Position', 'Error_Color'], axis=1)
        df = df.sort_values(['Condition', 'Electrode'])

        self.video_idxs = df['Condition'].unique()
        self.electrodes = df['Electrode'].unique()
        self.data = []

        for video_idx in self.video_idxs:
            seeg = torch.tensor((df[df['Condition'] == video_idx].iloc[:, 4:]).values, dtype=torch.float32)
            video_path = os.path.join(self.video_dir, f'mov{video_idx}.avi')
            video = process_video(video_path, self.video_processor_ckpt, self.num_frame_2_sample)
            self.data.append((seeg, video, torch.tensor(video_idx - 1, dtype=torch.float32)))

    def __getitem__(self, idx):
        seeg, video, video_idx = self.data[idx]

        # TODO: Remove the following line. It is only for testing.
        seeg_padding_mask = torch.zeros(seeg.shape[0], dtype=torch.bool)

        return seeg, seeg_padding_mask, video, video_idx

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    phase = 'Encoding'
    id = 'e0010GP'
    seeg_dir = '../data/dev/sEEG/'
    video_dir = '../data/dev/Movie Clips/'

    dataset = Dataset4Individual(id, phase, seeg_dir, video_dir)

    for i in range(len(dataset)):
        seeg, seeg_padding_mask, video, video_idx = dataset[i]
        print(seeg.shape, seeg_padding_mask.shape, video.shape, video_idx)
