import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import cv2 as cv


class MovieSEEGDataset(Dataset):
    def __init__(self, id, phase, seeg_dir, video_dir):
        self.id = id
        self.phase = phase
        self.seeg_dir = seeg_dir
        self.video_dir = video_dir

        df = pd.read_csv(os.path.join(self.seeg_dir, f'{self.id}_preprocessed_data.csv'))
        df = df[df['Phase'] == self.phase]
        df = df.drop(['Error_Position', 'Error_Color'], axis=1)
        df = df.sort_values(['Condition', 'Electrode'])

        self.video_idxs = df['Condition'].unique()
        self.electrodes = df['Electrode'].unique()
        self.data = []

        for video_idx in self.video_idxs:
            seeg = df[df['Condition'] == video_idx].iloc[:, 4:]
            cap = cv.VideoCapture(os.path.join(self.video_dir, f'mov{video_idx}.avi'))
            video = []
            while True:
                ret, frame = cap.read()
                if ret:
                    video.append(frame)
                else:
                    break
            video = np.array(video)
            self.data.append((seeg, video, video_idx - 1))

    def __getitem__(self, idx):
        seeg, video, video_idx = self.data[idx]
        return seeg, video, video_idx

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    phase = 'Encoding'
    id = 'e0010GP'
    seeg_dir = '../data/dev/sEEG/'
    video_dir = '../data/dev/Movie Clips/'

    dataset = MovieSEEGDataset(id, phase, seeg_dir, video_dir)

    for i in range(len(dataset)):
        seeg, video, video_idx = dataset[i]
        print(seeg.shape, video.shape, video_idx)

