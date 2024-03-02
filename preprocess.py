import av
import numpy as np
from transformers import VideoMAEImageProcessor
from transformers import AutoImageProcessor, AutoModel
import torch
from torchvision import transforms
from PIL import Image
import os


def get_frames(file_path):
    container = av.open(file_path)
    container.seek(0)
    return np.stack([frame.to_ndarray(format="rgb24") for i, frame in enumerate(container.decode(video=0))])


def videomae_preprocess(avi_path, ckpt="MCG-NJU/videomae-base"):
    image_processor = VideoMAEImageProcessor.from_pretrained(ckpt)
    # load avi files from greenbook_01 to greenbook_19
    video_data = []
    for i in range(1, 20):
        file_path = avi_path + f'/greenbook_{i:02d}.avi'
        video = get_frames(file_path)
        video_videomae = image_processor(list(video), return_tensors="pt")['pixel_values'][0]
        video_data.append(video_videomae.numpy())
        print(f'greenbook_{i:02d}.avi: {video_videomae.shape}')
    video_data = np.concatenate(video_data,axis=0)

    # split into 5s windows, drop the last window if it's less than 5s
    video_sr = 30.0
    window_size = int(5 * video_sr)
    n_windows = int(video_data.shape[0] // window_size)
    video_data = video_data[:n_windows * window_size]
    video_data = np.split(video_data, n_windows, axis=0)

    # save each element in the video_data list as a separate file
    # first make a folder called greenbook_videomae
    os.makedirs('greenbook_videomae', exist_ok=True)
    for i, window in enumerate(video_data):
        np.save(f'greenbook_videomae/greenbook_videomae_{i:02d}.npy', window)




def seeg_preprocess(seeg_path):
    seeg_contacts = np.load(seeg_path)
    seeg_sr = 1024.0
    # split into 5s windows, drop the last window if it's less than 5s
    window_size = int(5 * seeg_sr)
    n_windows = int(seeg_contacts.shape[1] // window_size)
    seeg_windows = np.split(seeg_contacts[:, :n_windows * window_size], n_windows, axis=1)

    # save each element in the seeg_windows list as a separate file
    # first make a folder called seeg
    os.makedirs('seeg', exist_ok=True)
    for i, window in enumerate(seeg_windows):
        np.save(f'seeg/seeg_{i:02d}.npy', window)


def dinos_preprocess(avi_path):
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),  # Resize the image
        transforms.CenterCrop(224),  # Crop the center of the image
        transforms.ToTensor(),  # Convert the image to a tensor and scale to [0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])

    frame_embeddings = []

    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    model.eval()

    # load avi files from greenbook_01 to greenbook_19
    for i in range(1, 20):
        file_path = avi_path + f'/greenbook_{i:02d}.avi'
        container = av.open(file_path)
        container.seek(0)
        for i, frame in enumerate(container.decode(video=0)):
            frame_rgb = frame.to_ndarray(format="rgb24")
            frame_preprocessed = transform(Image.fromarray(frame_rgb)).unsqueeze(0)
            with torch.no_grad():
                frame_embedding = model(frame_preprocessed)
            frame_embeddings.append(frame_embedding)
    frame_embeddings = torch.stack(frame_embeddings).squeeze()
    # split into 5s windows, drop the last window if it's less than 5s
    video_sr = 30.0
    window_size = int(5 * video_sr)
    n_windows = int(frame_embeddings.shape[0] // window_size)
    frame_embeddings = frame_embeddings[:n_windows * window_size]
    frame_embeddings = frame_embeddings.split(window_size, dim=0)

    # save each element in the video_data list as a separate file
    # first make a folder called greenbook_dinos
    os.makedirs('greenbook_dinos', exist_ok=True)
    for i, window in enumerate(frame_embeddings):
        torch.save(window, f'greenbook_dinos/greenbook_dinos_{i:02d}.pt')


if __name__ == "__main__":
    videomae_preprocess('/users/ycheng70/data/ycheng70/proj-brainstorm2024/data/green_book')
    seeg_preprocess('/users/ycheng70/data/ycheng70/proj-brainstorm2024/data/seeg_contacts.npy')
    dinos_preprocess('/users/ycheng70/data/ycheng70/proj-brainstorm2024/data/green_book')