import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import random_split, DataLoader


class Dataset4Individual(Dataset):
    def __init__(self, id, phases, seeg_dir):
        self.id = id
        self.phases = phases
        self.seeg_dir = seeg_dir

        assert os.path.exists(os.path.join(self.seeg_dir, f'{self.id}_preprocessed_data.csv'))
        df = pd.read_csv(os.path.join(self.seeg_dir, f'{self.id}_preprocessed_data.csv'))
        df = df.drop(['Error_Position', 'Error_Color'], axis=1)
        df = df.sort_values(['Condition', 'Electrode'])

        self.video_idxs = df['Condition'].unique()
        self.video_idxs = np.arange(1, 31, 1)
        self.electrodes = df['Electrode'].unique()
        if self.phases is None:
            self.phases = df['Phase'].unique()

        self.data = []
        for video_idx in self.video_idxs:
            for phase in self.phases:
                seeg = df[(df['Condition'] == video_idx) & (df['Phase'] == phase)].iloc[:, 4:].astype('float32')
                if not seeg.empty:
                    min_val = seeg.values.min()
                    max_val = seeg.values.max()
                    normalized_seeg = (seeg.values - min_val) / (max_val - min_val)
                    self.data.append((normalized_seeg, video_idx - 1, self.phases.index(phase)))

    def __getitem__(self, idx):
        seeg, video_idx, phase = self.data[idx]
        return seeg, video_idx, phase

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    phase = ['Encoding', 'SameDayRecall', 'NextDayRecall']
    id = 'e0010GP'
    seeg_dir = '../data/dev/sEEG/'
    dataset = Dataset4Individual(id, phase, seeg_dir)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=30, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=30, shuffle=False)

    # test the data loader
    for seeg, video_idx, phase in train_loader:
        print(seeg.shape)
        print(video_idx)
        print(phase)

    for seeg, video_idx, phase in val_loader:
        print(seeg.shape)
        print(video_idx)
        print(phase)
