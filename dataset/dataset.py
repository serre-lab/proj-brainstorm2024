import numpy as np
import glob
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """
    Custom dataset for the audio and SEEG data

    Parameters:
    - seeg_dir (str): path to the directory containing the sEEG data
    - video_dir (str): path to the directory containing the video data
    """
    def __init__(self, seeg_dir, video_dir, num_frame_2_sample=16):
        super(CustomDataset).__init__()
        # Load the sEEG
        self.seeg_data = []
        seeg_files = glob.glob(seeg_dir + '/*.npy')
        seeg_files.sort(key=lambda x: int(x.split('/')[-1].split('.')[0][5:]))
        self.seeg_files = seeg_files

        # Load and process the video data
        self.video_data = []
        self.num_frame_2_sample = num_frame_2_sample
        video_files = glob.glob(video_dir + '/*.npy')
        video_files.sort(key=lambda x: int(x.split('/')[-1].split('.')[0][19:]))
        self.video_files = video_files

        assert len(self.seeg_files) == len(self.video_files)

        self.total_num = len(self.video_files)

    def __getitem__(self, index):
        """
        Parameters:
        - index (int): index of the data to retrieve

        Returns:
        - video (torch.Tensor): the video data of shape (self.num_frame_2_sample, 3, 224, 224)
        - seeg (torch.Tensor): the sEEG data of shape (num_channels, 5120)
        """
        # Load and process the audio data
        video_file = self.video_files[index]
        video = np.load(video_file, allow_pickle=True)
        video = self.sample_frames(video)
        see_file = self.seeg_files[index]
        seeg = np.load(see_file, allow_pickle=True)
        return video.astype('float32'), seeg.astype('float32')

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
        offset = np.random.randint(0, 10)
        indices = np.linspace(offset, video_array.shape[0] - 10 + offset, num=self.num_frame_2_sample).astype(int)
        return video_array[indices]


if __name__ == '__main__':
    from util.experiment import get_args
    args = get_args()
    seeg_dir = args.seeg_dir
    video_dir = args.video_dir
    dataset = CustomDataset(seeg_dir, video_dir)
