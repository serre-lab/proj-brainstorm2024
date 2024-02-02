import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from util.data import process_video
from torch.utils.data import random_split, DataLoader
from torch.utils.data.sampler import BatchSampler
import numpy as np

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
        self.electrodes = df['Electrode'].unique()
        if self.phases is None:
            self.phases = df['Phase'].unique()

        self.data_by_condition = {idx: [] for idx in self.video_idxs}
        for video_idx in self.video_idxs:
            for phase in self.phases:
                seeg = df[(df['Condition'] == video_idx) & (df['Phase'] == phase)].iloc[:, 4:]
                if not seeg.empty:
                    self.data_by_condition[video_idx].append((seeg.values, video_idx - 1, self.phases.index(phase)))


    def __getitem__(self, idx):
        condition, data_idx = idx
        seeg, video_idx, phase = self.data_by_condition[condition][data_idx]
        return seeg, video_idx, phase

    def __len__(self):
        return sum(len(items) for items in self.data_by_condition.values())


class MovieSampler(BatchSampler):
    def __init__(self, dataset, batch_size=30):
        self.dataset = dataset
        self.batch_size = batch_size
        self.conditions = list(dataset.data_by_condition.keys())

    def __iter__(self):
        batch = []
        while len(batch) < self.batch_size:
            for condition in np.random.permutation(self.conditions):
                if len(self.dataset.data_by_condition[condition]) > 0:
                    batch.append((condition, np.random.randint(len(self.dataset.data_by_condition[condition]))))
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []

    def __len__(self):
        return int(np.ceil(len(self.dataset) / self.batch_size))

class MovieSampler2(BatchSampler):
    def __init__(self, dataset, batch_size=30, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.conditions = list(dataset.data_by_condition.keys())
        # Initialize a structure to track sampled items for each condition
        self.condition_indices = {condition: list(range(len(dataset.data_by_condition[condition]))) for condition in self.conditions}

    def __iter__(self):
        batch = []
        while True:
            np.random.shuffle(self.conditions)  # Shuffle conditions to ensure randomness
            for condition in self.conditions:
                if len(self.condition_indices[condition]) > 0:
                    # Randomly select an index to sample from this condition
                    index = np.random.choice(self.condition_indices[condition])
                    batch.append((condition, index))
                    # Remove the sampled index to avoid resampling
                    self.condition_indices[condition].remove(index)
                    if len(batch) == self.batch_size:
                        yield batch
                        batch = []
            # Check if there are enough items left to form a new batch; if not, and drop_last is False, yield what's left
            if not self.drop_last and len(batch) > 0 and all(len(indices) == 0 for indices in self.condition_indices.values()):
                yield batch
                break
            # If all items have been sampled, stop the iterator
            if all(len(indices) == 0 for indices in self.condition_indices.values()):
                break

    def __len__(self):
        total_samples = sum(len(items) for items in self.dataset.data_by_condition.values())
        if self.drop_last:
            return total_samples // self.batch_size
        else:
            return (total_samples + self.batch_size - 1) // self.batch_size




if __name__ == '__main__':
    phase = ['Encoding', 'SameDayRecall', 'NextDayRecall']
    id = 'e0010GP'
    seeg_dir = '../data/dev/sEEG/'

    dataset = Dataset4Individual(id, phase, seeg_dir)
    sampler = MovieSampler(dataset, batch_size=30)

    # test the dataloader
    data_loader = DataLoader(dataset, batch_sampler=sampler)

    for i, (seeg, video_idx, phase) in enumerate(data_loader):
        print(seeg.shape, video_idx, phase)
