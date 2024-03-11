import numpy as np
import glob
from torch.utils.data import Dataset
from util.data import Sampler


class BaseDataset(Dataset):
    def __init__(self, seeg_file, video_dir, video_file_prefix_len, time_window, sample_rate=1):
        # Load the sEEG data
        self.seeg_data = np.load(seeg_file)
        self.seeg_data = self.seeg_data[:, :self.seeg_data.shape[1] // (time_window * 1024) * (time_window * 1024)]\
            .reshape(84, -1, time_window * 1024).transpose(1, 0, 2).astype(np.float32)
        if sample_rate != 1:
            self.seeg_data = self.seeg_data.transpose(2, 0, 1)
            self.seeg_data = Sampler.sample(self.seeg_data, self.seeg_data.shape[0] // sample_rate, mode='even')
            self.seeg_data = self.seeg_data.transpose(1, 2, 0)

        # Load the video embeddings
        video_files = glob.glob(video_dir + '/*.npy')
        video_files.sort(key=lambda x: int(x.replace('\\', '/').split('/')[-1].split('.')[0][video_file_prefix_len:]))
        for video_file in video_files:
            video = np.load(video_file).astype(np.float32)
            self.video_data = video[None, :] if not hasattr(self, 'video_data') \
                else np.concatenate([self.video_data, video[None, :]], axis=0)

        min_len = min(self.seeg_data.shape[0], self.video_data.shape[0])
        self.seeg_data = self.seeg_data[:min_len]
        self.video_data = self.video_data[:min_len]
        print(f'Initialized dataset with {min_len} samples')

        self.total_num = min_len

    def __getitem__(self, index):
        """
        Parameters:
        - index (int): index of the data to retrieve

        Returns:
        - video (torch.Tensor): the video embedding data of shape(150, 768)
        - seeg (torch.Tensor): the sEEG data of shape (num_channels, 5120)
        """
        # Load and process the audio data
        video = self.video_data[index]
        seeg = self.seeg_data[index]
        return video, seeg

    def __len__(self):
        return self.total_num


class DinoDataset(BaseDataset):
    """
    Custom dataset for the Dino embeddings and sEEG data

    Parameters:
    - seeg_file (str): path to the sEEG data file
    - video_dir (str): path to the directory containing the Dino embeddings
    """
    def __init__(self, seeg_file, video_dir, time_window, sample_rate=1):
        dino_file_prefix = 'greenbook_dinos_'
        super().__init__(seeg_file, video_dir, len(dino_file_prefix), time_window, sample_rate)


class VideoMAEDataset(BaseDataset):
    """
    Custom dataset for the VideoMAE embeddings and sEEG data

    Parameters:
    - seeg_file (str): path to the sEEG data file
    - video_dir (str): path to the directory containing the VideoMAE embeddings
    """
    def __init__(self, seeg_file, video_dir, time_window, sample_rate=1):
        videomae_file_prefix = 'greenbook_videomae_'
        super().__init__(seeg_file, video_dir, len(videomae_file_prefix), time_window, sample_rate)


if __name__ == '__main__':
    seeg_file = '/gpfs/data/tserre/Shared/Brainstorm_2024/all_seeg_data.npy'
    dino_dir = '/gpfs/data/tserre/Shared/Brainstorm_2024/greenbook_dinos'
    time_window = 5
    dino_dataset = DinoDataset(seeg_file, dino_dir, time_window)
    for i in range(10):
        video, seeg = dino_dataset[i]
        assert video.shape == (150, 768)
        assert seeg.shape == (84, 5120)

    videomae_dir = '/gpfs/data/tserre/Shared/Brainstorm_2024/greenbook_videomae_features_2s'
    time_window = 2
    videomae_dataset = VideoMAEDataset(seeg_file, videomae_dir, time_window)
    for i in range(10):
        video, seeg = videomae_dataset[i]
        assert video.shape == (768, )
        assert seeg.shape == (84, 2048)

    dino_dir = '/gpfs/data/tserre/Shared/Brainstorm_2024/greenbook_dinos'
    time_window = 5
    sample_rate = 10
    dino_dataset = DinoDataset(seeg_file, dino_dir, time_window, sample_rate)
    for i in range(10):
        video, seeg = dino_dataset[i]
        assert video.shape == (150, 768)
        assert seeg.shape == (84, 512)
