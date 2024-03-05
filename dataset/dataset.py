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
        video_files = glob.glob(video_dir + '/*.pt')
        video_files.sort(key=lambda x: int(x.replace('\\', '/').split('/')[-1].split('.')[0][16:]))
        self.video_files = video_files

        assert len(self.seeg_files) == len(self.video_files)

        self.total_num = len(self.video_files)

    def __getitem__(self, index):
        """
        Parameters:
        - index (int): index of the data to retrieve

        Returns:
        - video (torch.Tensor): the video emebdding data of shape(150, 768)
        - seeg (torch.Tensor): the sEEG data of shape (num_channels, 5120)
        """
        # Load and process the audio data
        video_file = self.video_files[index]
        video = torch.load(video_file, map_location='cpu').float()
        see_file = self.seeg_files[index]
        seeg = np.load(see_file, allow_pickle=True).astype(np.float32)
        return video, seeg

    def __len__(self):
        return self.total_num


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
