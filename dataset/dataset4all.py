import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from util.data import process_video


class Dataset4All(Dataset):
    def __init__(self, ids, phases, seeg_dir, video_dir, video_processor_ckpt="MCG-NJU/videomae-base",
                 num_frame_2_sample=16):
        self.ids = ids
        self.phases = phases
        self.seeg_dir = seeg_dir
        self.video_dir = video_dir
        self.video_processor_ckpt = video_processor_ckpt
        self.num_frame_2_sample = num_frame_2_sample

        df_list = []
        for id in self.ids:
            assert os.path.exists(os.path.join(self.seeg_dir, f'{id}_preprocessed_data.csv'))
            df = pd.read_csv(os.path.join(self.seeg_dir, f'{id}_preprocessed_data.csv'))
            df = df.drop(['Error_Position', 'Error_Color'], axis=1)
            df_list.append(df)
            df_all = pd.concat(df_list).sort_values(['Condition', 'Electrode'])

        self.video_idxs = df_all['Condition'].unique()
        self.electrodes = df_all['Electrode'].unique()
        if self.ids is None:
            self.ids = df_all['Participant_ID'].unique().tolist()
        if self.phases is None:
            self.phases = df_all['Phase'].unique()

        self.data = []
        for video_idx in self.video_idxs:
            video_path = os.path.join(self.video_dir, f'mov{video_idx}.avi')
            assert os.path.exists(video_path)
            video = process_video(video_path, self.video_processor_ckpt, self.num_frame_2_sample)
            for participant in self.ids:
                for phase in self.phases:
                    seeg = df_all[(df_all['Condition'] == video_idx) & (df_all['Participant_ID'] == participant) & (df_all['Phase'] == phase)].iloc[:, 4:]
                    seeg = seeg.reindex(self.electrodes, axis=0, fill_value=0)
                    seeg = torch.tensor(seeg.values, dtype=torch.float32)
                    seeg_mask = seeg == 0
                    self.data.append((seeg, seeg_mask, video, torch.tensor(video_idx - 1, dtype=torch.float32),
                                      self.phases.index(phase), self.ids.index(participant)))

    def __getitem__(self, idx):
        seeg, seeg_mask, video, video_idx, phase, id = self.data[idx]
        return seeg, seeg_mask, video, video_idx, phase, id

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    phases = ['Encoding', 'SameDayRecall', 'NextDayRecall']
    ids = ['e0010GP', 'e0011XQ', 'e0013LW', 'e0015TJ', 'e0016YR', 'e0017MC', 'e0019VQ', 'e0020JA', 'e0022ZG', 'e0024DV']
    seeg_dir = '../data/dev/sEEG/'
    video_dir = '../data/dev/Movie Clips/'

    dataset = Dataset4All(ids, phases, seeg_dir, video_dir)

    for i in range(len(dataset)):
        seeg, seeg_mask, video, video_idx, phase, id = dataset[i]
        print(seeg.shape, seeg_mask.shape, video.shape, video_idx, phase, id)

