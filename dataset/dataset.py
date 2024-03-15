import numpy as np
import glob
from torch.utils.data import Dataset
from util.data import Sampler
from util.data import get_scene_timestamp


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
        # Load the sEEG data
        seeg_data = np.load(seeg_file).astype(np.float32)

        # Resplit the seeg data according to the timestamps
        self.max_seeg_length = 6865
        self.seeg_data = []
        for timestamp in timestamps:
            start, end = timestamp
            start = round((start[0] * 3600 + start[1] * 60 + start[2] + start[3] / 1000) * 1024)
            end = round((end[0] * 3600 + end[1] * 60 + end[2] + end[3] / 1000) * 1024)
            seeg = seeg_data[:, start:end]
            self.seeg_data.append(seeg)

        # Load the video embeddings
        self.video_max_length = 201
        video_file_prefix_len = len('greenbook_dinos_')
        video_files = glob.glob(video_dir + '/*.npy')
        video_files.sort(key=lambda x: int(x.replace('\\', '/').split('/')[-1].split('.')[0][video_file_prefix_len:]))
        for video_file in video_files:
            video = np.load(video_file).astype(np.float32)
            video_mask = np.zeros((self.video_max_length, 1))
            video_mask[video.shape[0]:] = True
            self.video_masks = video_mask[None, :] if not hasattr(self, 'video_masks') \
                else np.concatenate([self.video_masks, video_mask[None, :]], axis=0)
            video = np.pad(video, ((0, 201 - video.shape[0]), (0, 0)))
            self.video_data = video[None, :] if not hasattr(self, 'video_data') \
                else np.concatenate([self.video_data, video[None, :]], axis=0)

        min_len = min(len(self.seeg_data), self.video_data.shape[0])
        self.seeg_data = self.seeg_data[:min_len]
        self.video_data = self.video_data[:min_len]
        print(f'Initialized dataset with {min_len} samples')

        self.total_num = min_len

    def __getitem__(self, index):
        video = self.video_data[index]
        video_mask = np.zeros((self.video_max_length, 1))
        video_mask[video.shape[0]:] = True
        video = np.pad(video, ((0, self.video_max_length - video.shape[0]), (0, 0)))

        seeg = self.seeg_data[index]
        seeg_mask = np.zeros((self.seeg_max_length, 1))
        seeg_mask[seeg.shape[0]:] = True
        seeg = np.pad(seeg, ((0, self.seeg_max_length - seeg.shape[0]), (0, 0)))
        return video, video_mask, seeg, seeg_mask


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
    file_path = '/gpfs/data/tserre/Shared/Brainstorm_2024/GreenBook.txt'
    timestamps = get_scene_timestamp(file_path)
    dino_scene_dataset = DinoSceneDataset(seeg_file, dino_dir, timestamps)
    for i in range(10):
        video, video_mask, seeg, seeg_mask = dino_scene_dataset[i]
        assert video.shape == (201, 768)
        assert video_mask.shape == (201, 1)
        first_nonzero = np.argmax(video_mask)
        assert np.all(video[first_nonzero:] != 0)
        assert np.all(video[:first_nonzero] == 0)

        assert seeg.shape == (84, dino_scene_dataset.seeg_max_length)
        assert seeg_mask.shape == (dino_scene_dataset.seeg_max_length, 1)
        first_nonzero = np.argmax(seeg_mask)
        for j in range(84):
            assert np.all(seeg[j, first_nonzero:] != 0)
            assert np.all(seeg[j, :first_nonzero] == 0)

