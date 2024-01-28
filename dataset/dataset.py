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

        df = pd.read_csv(os.path.join(self.seeg_dir, self.id + "_preprocessed_data.csv"))
        df = df[df['Phase'] == self.phase]
        df = df.drop(['Error_Position', 'Error_Color'], axis=1)
        df = df.sort_values(['Condition', 'Electrode'])

        self.movie_clips = df['Condition'].unique()
        self.electrodes = df['Electrode'].unique()
        self.video_seeg = []
        for movie_clip in self.movie_clips:
            df_movie_clip = df[df['Condition'] == movie_clip].iloc[:, 4:]
            cap = cv.VideoCapture(self.video_dir + str(movie_clip) + ".avi")
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


if __name__ == '__main__':
    phase = 'Encoding'
    id = 'e0010GP'
    seeg_dir = '../data/dev/sEEG/'
    video_dir = '../data/dev/Movie Clips/'

    dataset = MovieSEEGDataset(id, phase, seeg_dir, video_dir)

    for i in range(len(dataset)):
        movie_clip, df_movie_clip, frames = dataset[i]
        print(movie_clip, df_movie_clip, frames)

