import cv2
import numpy as np
from transformers import VideoMAEImageProcessor
import torch


def get_frames(file_path):
    cap = cv2.VideoCapture(file_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert the frame format from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    return np.stack(frames)


def videomae_preprocess(avi_path, ckpt="MCG-NJU/videomae-base"):
    image_processor = VideoMAEImageProcessor.from_pretrained(ckpt)
    # load avi files from greenbook_01 to greenbook_19
    video_data = []
    for i in range(1, 20):
        file_path = avi_path + f'\greenbook_{i:02d}.avi'
        video = get_frames(file_path)
        video_videomae = image_processor(np.array(video), return_tensors="pt")['pixel_values'][0]
        video_data.append(video_videomae.numpy())
        print(f'greenbook_{i:02d}.avi: {video_videomae.shape}')
    video_data = np.array(video_data, dtype=np.float32)

    # split into 5s windows, drop the last window if it's less than 5s
    video_sr = 30.0
    window_size = int(5 * video_sr)
    n_windows = int(video_data.shape[1] // window_size)
    video_data = video_data[:, :n_windows * window_size]
    video_data = np.split(video_data, n_windows, axis=1)
    video_data = np.stack(video_data)
    print(video_data.shape)
    np.save('greenbook_videomae.npy', video_data)


def seeg_preprocess(seeg_path):
    seeg_contacts = np.load(seeg_path)
    seeg_sr = 1024.0
    # split into 5s windows, drop the last window if it's less than 5s
    window_size = int(5 * seeg_sr)
    n_windows = int(seeg_contacts.shape[1] // window_size)
    seeg_windows = np.split(seeg_contacts[:, :n_windows * window_size], n_windows, axis=1)
    seeg_windows = np.stack(seeg_windows)
    print(seeg_windows.shape)
    np.save('seeg_split.npy', seeg_windows)


def dinos_preprocess():
    pass


if __name__ == "__main__":
    videomae_preprocess(
        r'C:\Users\ycheng70\OneDrive - Brown University\Documents\proj-subtitles-decoding\data\green_book')
    seeg_preprocess(
        r'C:\Users\ycheng70\OneDrive - Brown University\Documents\proj-subtitles-decoding\data\seeg_contacts.npy')
