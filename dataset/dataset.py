import numpy as np
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """
    Custom dataset for the audio and SEEG data

    Parameters:
    - data_file (str): path to the .npy file containing the data
    """
    def __init__(self, data_file):
        super(CustomDataset).__init__()
        # Load the data
        data = np.load(data_file, allow_pickle=True)
        self.video_data = data.item()['video']
        self.seeg_data = data.item()['seeg']
        self.total_num = len(self.video_data)

    def __getitem__(self, index):
        """
        Parameters:
        - index (int): index of the data to retrieve

        Returns:
        - video (torch.Tensor): the video data of shape (16, 3, 224, 224)
        - seeg (torch.Tensor): the sEEG data of shape (num_channels, 5120)
        """
        # Load and process the audio data
        video = self.video_data[index]
        seeg = self.seeg_data[index]
        return video, seeg

    def __len__(self):
        return self.total_num