import numpy as np
import pandas as pd
import torch
import torch.nn as
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2 as cv

class MOVIESEEGDataset(Dataset):
    def __init__(self, args):

        self.ID = args.ID
        self.phase = args.phase
        self.seeg_path = "/oscar/data/brainstorm-ws/seeg_data/Memory Task Data/Epilepsy/Monitoring/"
        self.video_path = "/oscar/data/brainstorm-ws/seeg_data/Movie Clips/"
        df = pd.read_csv(self.seeg_path + self.ID + "_preprocessed_data.csv")
        df = df[df['Phase'] == self.phase]
        df = df.drop(['Error_Position', 'Error_Color'], axis=1)
        df = df.sort_values(['Condition', 'Electrode'])
        self.movie_clips = df['Condition'].unique()
        self.electrodes = df['Electrode'].unique()
        self.video_seeg = []
        for movie_clip in self.movie_clips:
            df_movie_clip = df[df['Condition'] == movie_clip].iloc[:, 4:]
            cap = cv.VideoCapture(self.video_path + str(movie_clip) + ".avi")
            frames = []
            while True:
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
                else:
                    break
            frames = np.array(frames)
            self.video_seeg.append((movie_clip, df_movie_clip, frames))

    def __len__(self):
        return len(self.video_seeg)

    def __getitem__(self, idx):
        movie_clip, df_movie_clip, frames = self.video_seeg[idx]
        return movie_clip, df_movie_clip, frames