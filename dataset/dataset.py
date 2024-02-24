import numpy as np
import glob
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """
    Custom dataset for the audio and SEEG data

    Parameters:
    - seeg_file (str): path to the sEEG data file
    - video_dir (str): path to the directory containing the video data
    """
    def __init__(self, seeg_file, video_dir, num_frame_2_sample=16):
        super(CustomDataset).__init__()
        # Load the sEEG
        self.seeg_data = np.array(np.load(seeg_file, allow_pickle=True).item()['seeg'])

        # Load and process the video data
        self.video_data = []
        self.num_frame_2_sample = num_frame_2_sample
        video_files = glob.glob(video_dir + '/*.npy')
        video_files.sort(key=lambda x: int(x.split('/')[-1].split('.')[0][2:]))
        for video_file in video_files:
            video = np.load(video_file, allow_pickle=True)
            video = self.sample_frames(video)
            self.video_data.append(video)
        self.video_data = np.array(self.video_data)

        self.total_num = len(self.video_data)

    def __getitem__(self, index):
        """
        Parameters:
        - index (int): index of the data to retrieve

        Returns:
        - video (torch.Tensor): the video data of shape (self.num_frame_2_sample, 3, 224, 224)
        - seeg (torch.Tensor): the sEEG data of shape (num_channels, 5120)
        """
        # Load and process the audio data
        video = self.video_data[index]
        seeg = self.seeg_data[index]
        return video, seeg

    def __len__(self):
        return self.total_num

    def sample_frames(self, video_array):
        """
        Sample a given number of frames from the video array evenly.
        Parameters:
        - video_array (np.ndarray): the video data of shape (num_frames, 3, 224, 224)

        Returns:
        - video (np.ndarray): the sampled video data of shape (num_frame_2_sample, 3, 224, 224)
        """
        indices = np.linspace(0, video_array.shape[0] - 1, num=self.num_frame_2_sample).astype(int)
        return video_array[indices]
