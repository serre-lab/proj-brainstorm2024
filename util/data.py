import av
import torch
import glob
import os
from tqdm import tqdm
import numpy as np
from util.model import VideoMAECls


def extract_videomae_features(frame_dir, output_dir, num_frame_2_sample=16, interval=1):
    """
    Extract VideoMAE features from the preprocessed video frames
    Parameters:
    - frame_dir (str): the directory containing the preprocessed video frames
    - output_dir (str): the directory to save the extracted features
    - num_frame_2_sample (int): the number of frames to sample from each video
    - interval (int): the interval between the sampled frames
    Returns:
    - features (np.ndarray): the extracted features
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VideoMAECls.from_pretrained("sayakpaul/videomae-base-finetuned-kinetics-finetuned-ucf101-subset").to(device)
    frame_file_prefix = 'greenbook_videomae_'
    frame_files = glob.glob(frame_dir + '/*.npy')
    frame_files.sort(key=lambda x: int(x.replace('\\', '/').split('/')[-1].split('.')[0][len(frame_file_prefix):]))

    counter = 0
    for i in tqdm(range(len(frame_files))):
        if i % 2 != 0:
            continue
        if i + 1 >= len(frame_files):
            frames = np.load(frame_files[i])[:120].reshape(2, 60, 3, 224, 224)
        else:
            frames_1 = np.load(frame_files[i])
            frames_2 = np.load(frame_files[i + 1])
            frames = np.concatenate([frames_1, frames_2], axis=0).reshape(5, 60, 3, 224, 224)

        inputs = None
        for input in frames:
            # Sample the middlemost frames
            input = Sampler.sample(input, num_frame_2_sample, mode='dense', interval=interval,
                                   start_idx=30 - num_frame_2_sample // 2)
            inputs = input[None, :] if inputs is None else np.concatenate([inputs, input[None, :]], axis=0)

        inputs = torch.tensor(inputs).to(device)
        with torch.no_grad():
            features = model(inputs).cpu().numpy()

        for feature in features:
            np.save(os.path.join(output_dir, f'greenbook_videomae_{counter}.npy'), feature)
            counter += 1


def resplit_dino_features(dino_dir, output_dir):
    """
    Resplit the dino features into 2s windows
    Parameters:
    - dino_dir (str): the directory containing the dino features
    - output_dir (str): the directory to save the extracted features
    Returns:
    - None
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    frame_file_prefix = 'greenbook_dinos_'
    frame_files = glob.glob(dino_dir + '/*.npy')
    frame_files.sort(key=lambda x: int(x.replace('\\', '/').split('/')[-1].split('.')[0][len(frame_file_prefix):]))

    counter = 0
    for i in tqdm(range(len(frame_files))):
        if i % 2 != 0:
            continue
        if i + 1 >= len(frame_files):
            frames = np.load(frame_files[i])[:120].reshape(2, 60, 768)
        else:
            frames_1 = np.load(frame_files[i])
            frames_2 = np.load(frame_files[i + 1])
            frames = np.concatenate([frames_1, frames_2], axis=0).reshape(5, 60, 768)

        for frame in frames:
            np.save(os.path.join(output_dir, f'greenbook_dinos_{counter}.npy'), frame)
            counter += 1


class Sampler:
    @staticmethod
    def sample(source, num_2_sample, mode='even', **kwargs):
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
            return Sampler._sample_even(source, num_2_sample)
        elif mode == 'dense':
            interval = kwargs.get('interval', None)
            if interval is None:
                raise ValueError("Interval must be provided for dense sampling")
            start_idx = kwargs.get('start_idx', None)
            return Sampler._sample_dense(source, num_2_sample, interval, start_idx)

    @staticmethod
    def _sample_even(source, num_2_sample):
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

    @staticmethod
    def _sample_dense(source, num_2_sample, interval, start_idx=None):
        """
        Sample num_2_sample units from the source densely. Random offsets are used to sample the units if
        len(source) - interval * (num_2_sample - 1) > 0
        Parameters:
        - source (np.ndarray): the source to sample from
        - num_2_sample (int): the number of units to sample
        - interval (int): the interval between the sampled units
        - start_idx (int): the starting index of the sampling, optional
        Returns:
        - sample (np.ndarray): the sampled units
        """
        total_len = source.shape[0]
        dof = total_len - interval * (num_2_sample - 1)
        if dof < 0:
            raise ValueError("The interval is too large for the number of units to sample")

        if start_idx:
            offsets = start_idx
        else:
            if dof == 0:
                offsets = 0
            else:
                offsets = np.random.randint(0, dof)
        idxs = np.arange(offsets, offsets + interval * (num_2_sample - 1) + 1, interval)
        return source[idxs]


def parse_timestamp(ts):
    """
    Parse a timestamp in the subtitle file
    :param ts: the timestamp
    :return: the parsed timestamp
    """
    h, m, s_ms = ts.split(':')
    s, ms = s_ms.split(',')
    return int(h), int(m), int(s), int(ms)


def get_scene_timestamp(file_path):
    """
    Get the scene timestamps from the subtitle file
    Parameters:
    - file (str): the subtitle file
    Returns:
    - timestamps (list): the scene timestamps, each element is a tuple (start, end)
    """
    timestamps = []
    with open(file_path, 'r', encoding='utf-8-sig') as file:
        for line in file:
            if '-->' in line:
                start_ts, end_ts = line.strip().split(' --> ')
                start_h, start_m, start_s, start_ms = parse_timestamp(start_ts)
                end_h, end_m, end_s, end_ms = parse_timestamp(end_ts)
                timestamps.append(((start_h, start_m, start_s, start_ms), (end_h, end_m, end_s, end_ms)))
    return timestamps


def resplit_dino_by_timestamp(dino_dir, timestamps, output_dir):
    """
    Resplit the dino features by the scene timestamps
    Parameters:
    - dino_dir (str): the directory containing the dino features
    - timestamps (list): the scene timestamps
    - output_dir (str): the directory to save the extracted features
    Returns:
    - None
    """
    frame_file_prefix = 'greenbook_dinos_'
    frame_files = glob.glob(dino_dir + '/*.npy')
    frame_files.sort(key=lambda x: int(x.replace('\\', '/').split('/')[-1].split('.')[0][len(frame_file_prefix):]))

    features = []
    for file in frame_files:
        feature = np.load(file)
        features.append(feature)

    features = np.array(features)
    for i, timestamp in tqdm(enumerate(timestamps)):
        start, end = timestamp
        start_frame = round((start[0] * 3600 + start[1] * 60 + start[2] + start[3] / 1000) * 30)
        end_frame = round((end[0] * 3600 + end[1] * 60 + end[2] + end[3] / 1000) * 30)
        feature = features[start_frame:end_frame]
        np.save(os.path.join(output_dir, f'greenbook_dinos_{i}.npy'), feature)


if __name__ == '__main__':
    # frame_dir = '/gpfs/data/tserre/Shared/Brainstorm_2024/greenbook_videomae_preprocessed_frames'
    # output_dir = '/gpfs/data/tserre/Shared/Brainstorm_2024/greenbook_videomae_features_2s'
    # extract_videomae_features(frame_dir, output_dir, num_frame_2_sample=16, interval=1)

    # frame_dir = '/gpfs/data/tserre/Shared/Brainstorm_2024/greenbook_dinos'
    # output_dir = '/gpfs/data/tserre/Shared/Brainstorm_2024/greenbook_dinos_2s'
    # extract_dino_features(frame_dir, output_dir, num_frame_2_sample=16, interval=1)

    file_path = '/gpfs/data/tserre/Shared/Brainstorm_2024/GreenBook.txt'
    timestamps = get_scene_timestamp(file_path)
