import av
import numpy as np
from transformers import VideoMAEImageProcessor
import torch
from torchvision import transforms
from PIL import Image
import os
from scipy.signal import iirnotch, filtfilt, butter


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


def apply_filters(data, fs, notch_freq=60.0, quality_factor=30, lowcut=0.1, highcut=50.0):
    """
    Apply notch and bandpass filters to the data.

    Parameters:
    - data: numpy array containing the SEEG data.
    - fs: Sampling frequency in Hz.
    - notch_freq: Frequency to be notched out.
    - quality_factor: Quality factor for the notch filter.
    - lowcut: Lower frequency bound for the bandpass filter.
    - highcut: Upper frequency bound for the bandpass filter.

    Returns:
    - filtered_data: The filtered SEEG data.
    """

    # Notch filter
    b_notch, a_notch = iirnotch(notch_freq, quality_factor, fs)
    data = filtfilt(b_notch, a_notch, data)

    # Bandpass filter
    b_bandpass, a_bandpass = butter(N=4, Wn=[lowcut, highcut], btype='band', fs=fs)
    filtered_data = filtfilt(b_bandpass, a_bandpass, data)

    return filtered_data


# Function to apply the filters to each channel of the data
def filter_all_channels(data, fs):
    # Assuming data shape is (channels, samples)
    filtered_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        filtered_data[i] = apply_filters(data[i], fs)
    return filtered_data


def filter_all_seeg(directory,new_directory):
    files = os.listdir(directory)  # List all files in the directory
    
    filtered_seeg_array = []  # This list will hold the filtered data arrays if needed
    
    for file in files:
        if file.endswith('.npy'):  # Ensure we're processing .npy files
            file_path = os.path.join(directory, file)
            data = np.load(file_path)
            
            fs = 1024  # Sampling frequency
            filtered_seeg_data = apply_filters(data, fs)
            
            filtered_seeg_array.append(filtered_seeg_data)  # Optional: collect filtered data
            
            # Generate a new filename for the filtered data
            filtered_file_path = os.path.join(new_directory, 'filtered_' + file)
            
            # Save the filtered data with the new filename
            np.save(filtered_file_path, filtered_seeg_data)


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


def dino_reprocess(avi_path):
    # load back each embedding
    # from greenbook_dinos_00.pt to greenbook_dinos_1468.pt
    # save each embedding as a separate file into numpy
    os.makedirs('greenbook_dinos', exist_ok=True)
    for i in range(1469):
        file_path = avi_path + f'/greenbook_dinos/greenbook_dinos_{i:02d}.pt'
        frame_embedding = torch.load(file_path)
        frame_embedding = frame_embedding.numpy()
        np.save(f'greenbook_dinos/greenbook_dinos_{i:02d}.npy', frame_embedding)


if __name__ == "__main__":
    # videomae_preprocess('/users/ycheng70/data/ycheng70/proj-brainstorm2024/data/green_book')
    # seeg_preprocess('/users/ycheng70/data/ycheng70/proj-brainstorm2024/data/seeg_contacts.npy')
    # dinos_preprocess('/users/ycheng70/data/ycheng70/proj-brainstorm2024/data/green_book')
    #
    # directory = '/gpfs/data/tserre/Shared/Brainstorm_2024/seeg/'
    # new_directory = '/gpfs/data/tserre/Shared/Brainstorm_2024/cleaned_seeg/'
    # filter_all_seeg(directory,new_directory)
    #
    # dino_reprocess('/users/ycheng70/data/ycheng70/proj-brainstorm2024/')
    pass
