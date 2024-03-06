import numpy as np
import glob
from torch.utils.data import Dataset
import torch


class CustomDataset(Dataset):
    """
    Custom dataset for the video frame embeddings and SEEG data

    Parameters:
    - seeg_dir (str): path to the directory containing the sEEG data
    - video_dir (str): path to the directory containing the video frame embeddings
    """
    def __init__(self, seeg_dir, video_dir, num_frame_2_sample=16):
        super(CustomDataset).__init__()
        # Load the sEEG, where each file is named seeg_XXX.npy
        self.seeg_data = []
        seeg_files = glob.glob(seeg_dir + '/*.npy')
        seeg_files.sort(key=lambda x: int(x.replace('\\', '/').split('/')[-1].split('.')[0][5:]))
        self.seeg_files = seeg_files

        # Load and process the video data
        self.video_data = []
        video_files = glob.glob(video_dir + '/*.npy')
        video_files.sort(key=lambda x: int(x.replace('\\', '/').split('/')[-1].split('.')[0][16:]))
        self.video_files = video_files

        assert len(self.seeg_files) == len(self.video_files)

        self.total_num = len(self.video_files)

    def __getitem__(self, index):
        """
        Parameters:
        - index (int): index of the data to retrieve

        Returns:
        - video (torch.Tensor): the video embedding data of shape(150, 768)
        - seeg (torch.Tensor): the sEEG data of shape (num_channels, 5120)
        """
        # Load and process the audio data
        video_file = self.video_files[index]
        video = np.load(video_file, allow_pickle=True).astype(np.float32)
        see_file = self.seeg_files[index]
        seeg = np.load(see_file, allow_pickle=True).astype(np.float32)
        return video, seeg

    def __len__(self):
        return self.total_num

    def sample(self, source, num_2_sample, mode='even', **kwargs):
        """
        Sample num_2_sample units from the source
        Parameters:
        - source (list): the source to sample from
        - num_2_sample (int): the number of units to sample
        - mode (str): the sampling mode, either 'even' or 'dense'
        - kwargs (dict): additional arguments for the sampling mode
        Returns:
        - sample (list): the sampled units
        """
        if mode == 'even':
            return self._sample_even(source, num_2_sample)
        elif mode == 'dense':
            interval = kwargs.get('interval', None)
            if interval is None:
                raise ValueError("Interval must be provided for dense sampling")
            return self._sample_dense(source, num_2_sample, interval)

    def _sample_even(self, source, num_2_sample):
        """
        Sample num_2_sample units from the source evenly. Random offsets are used to sample the units if
        len(source) % num_2_sample != 0
        Parameters:
        - source (np.ndarray): the source to sample from
        - num_2_sample (int): the number of units to sample
        Returns:
        - sample (np.ndarray): the sampled units
        """
        total_len = source.shape[0]
        dof = total_len % num_2_sample

        if dof == 0:
            offsets = 0
        else:
            offsets = np.random.randint(0, dof)
        idxs = np.linspace(offsets, total_len - dof + offsets - 1, num_2_sample).astype(int)
        return source[idxs]

    def _sample_dense(self, source, num_2_sample, interval):
        """
        Sample num_2_sample units from the source densely. Random offsets are used to sample the units if
        len(source) - interval * (num_2_sample - 1) > 0
        Parameters:
        - source (np.ndarray): the source to sample from
        - num_2_sample (int): the number of units to sample
        - interval (int): the interval between the sampled units
        Returns:
        - sample (np.ndarray): the sampled units
        """
        total_len = source.shape[0]
        dof = total_len - interval * (num_2_sample - 1)
        if dof < 0:
            raise ValueError("The interval is too large for the number of units to sample")

        if dof == 0:
            offsets = 0
        else:
            offsets = np.random.randint(0, dof)
        idxs = np.arange(offsets, offsets + interval * (num_2_sample - 1) + 1, interval)
        return source[idxs]


if __name__ == '__main__':
    from util.experiment import get_args
    args = get_args()
    args.seeg_dir = '../data/seeg'
    args.video_dir = '../data/greenbook_dinos'
    seeg_dir = args.seeg_dir
    video_dir = args.video_dir
    dataset = CustomDataset(seeg_dir, video_dir)
    # test the dataset
    for i in range(10):
        video, seeg = dataset[i]
        # print the memory usage of video and seeg in GB
        print(video.shape, seeg.shape)
