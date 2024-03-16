import numpy as np
import glob
from torch.utils.data import Dataset
from util.data import Sampler
from util.data import get_scene_timestamp
from tqdm import tqdm


class BaseDataset(Dataset):
    def __init__(self, seeg_file, video_dir, video_file_prefix_len, time_window, sample_rate=1):
        # Load the sEEG data
        self.seeg_data = np.load(seeg_file)
        self.seeg_data = self.seeg_data[:, :self.seeg_data.shape[1] // (time_window * 1024) * (time_window * 1024)]\
            .reshape(84, -1, time_window * 1024).transpose(1, 0, 2).astype(np.float32)

        self.num_frame_2_sample = self.seeg_data.shape[2] // sample_rate

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
        - video (np.ndarray): the video data of shape (150, 768)
        - seeg (np.ndarray): the sEEG data of shape (num_channels, num_frame_2_sample)
        """
        # Load and process the audio data
        video = self.video_data[index]
        seeg = self.seeg_data[index]
        seeg = seeg.transpose(1, 0)
        seeg = Sampler.sample(seeg, self.num_frame_2_sample, mode='even')
        seeg = seeg.transpose(1, 0)
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


class DinoSceneDataset(Dataset):
    def __init__(self, seeg_file, video_dir, timestamps):
        self.timestamps = timestamps

        # Load the sEEG data
        # self.max_seeg_length = 6865
        self.seeg_data = np.load(seeg_file).astype(np.float32)

        # Load the video embeddings
        # self.max_video_length = 201
        video_file_prefix_len = len('greenbook_dinos_')
        self.video_files = glob.glob(video_dir + '/*.npy')
        self.video_files.sort(key=lambda x: int(x.replace('\\', '/').split('/')[-1].split('.')[0][video_file_prefix_len:]))

        self.total_num = len(timestamps)

        print(f'Initialized dataset with {self.total_num} samples')

    def __getitem__(self, index):
        video_file = self.video_files[index]
        video = np.load(video_file).astype(np.float32)
        # video_mask = np.zeros((self.max_video_length,)).astype(bool)
        # video_mask[video.shape[0]:] = True
        video_mask = None
        # video = np.pad(video, ((0, self.max_video_length - video.shape[0]), (0, 0)))

        start, end = self.timestamps[index]
        start = round((start[0] * 3600 + start[1] * 60 + start[2] + start[3] / 1000) * 1024)
        end = round((end[0] * 3600 + end[1] * 60 + end[2] + end[3] / 1000) * 1024)
        seeg = self.seeg_data[:, start:end]
        # seeg_mask = np.zeros((self.max_seeg_length,)).astype(bool)
        # seeg_mask[seeg.shape[1]:] = True
        seeg_mask = None
        # seeg = np.pad(seeg, ((0, 0), (0, self.max_seeg_length - seeg.shape[1])))

        return video, video_mask, seeg, seeg_mask

    def __len__(self):
        return self.total_num


if __name__ == '__main__':
    seeg_file = '/gpfs/data/tserre/Shared/Brainstorm_2024/all_seeg_data.npy'
    # dino_dir = '/gpfs/data/tserre/Shared/Brainstorm_2024/greenbook_dinos'
    # time_window = 5
    # dino_dataset = DinoDataset(seeg_file, dino_dir, time_window)
    # for i in range(10):
    #     video, seeg = dino_dataset[i]
    #     assert video.shape == (150, 768)
    #     assert seeg.shape == (84, 5120)
    #
    # videomae_dir = '/gpfs/data/tserre/Shared/Brainstorm_2024/greenbook_videomae_features_2s'
    # time_window = 2
    # videomae_dataset = VideoMAEDataset(seeg_file, videomae_dir, time_window)
    # for i in range(10):
    #     video, seeg = videomae_dataset[i]
    #     assert video.shape == (768, )
    #     assert seeg.shape == (84, 2048)
    #
    # dino_dir = '/gpfs/data/tserre/Shared/Brainstorm_2024/greenbook_dinos'
    # time_window = 5
    # sample_rate = 10
    # dino_dataset = DinoDataset(seeg_file, dino_dir, time_window, sample_rate)
    # for i in range(10):
    #     video, seeg = dino_dataset[i]
    #     assert video.shape == (150, 768)
    #     assert seeg.shape == (84, 512)

    dino_dir = '/gpfs/data/tserre/Shared/Brainstorm_2024/greenbook_dinos_scenes'
    file_path = '/gpfs/data/tserre/Shared/Brainstorm_2024/Green Book.srt'
    timestamps = get_scene_timestamp(file_path)
    dino_scene_dataset = DinoSceneDataset(seeg_file, dino_dir, timestamps)
    # for i in range(10):
    #     video, video_mask, seeg, seeg_mask = dino_scene_dataset[i]
    #     assert video.shape == (201, 768)
    #     assert video_mask.shape == (201,)
    #     first_nonzero = np.argmax(video_mask)
    #     assert video[first_nonzero - 1, -1] != 0
    #     assert np.all(video[first_nonzero:, :] == 0)
    #
    #     assert seeg.shape == (84, 6865)
    #     assert seeg_mask.shape == (6865,)
    #     first_nonzero = np.argmax(seeg_mask)
    #     assert seeg[-1, first_nonzero - 1] != 0
    #     assert np.all(seeg[:, first_nonzero:] == 0)
    max_seeg_length = 0
    max_video_length = 0
    for i in range(len(dino_scene_dataset)):
        video, _, seeg, _ = dino_scene_dataset[i]
        max_seeg_length = max(max_seeg_length, seeg.shape[1])
        max_video_length = max(max_video_length, video.shape[0])
    print(max_seeg_length)
    print(max_video_length)

