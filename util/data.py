import av
import numpy as np
from transformers import VideoMAEImageProcessor


# The following function is adapted from the Hugging Face Transformers documentation.
# Source: Transformers: VideoMAEModel Documentation
# URL: https://huggingface.co/docs/transformers/model_doc/videomae#transformers.VideoMAEModel.forward.example
def get_frames(container, indices):
    """
    Get frames of given indices from a PyAV container.

    Parameters:
    - container (`av.container.input.InputContainer`): PyAV container.
    - indices (`List[int]`): List of frame indices to decode.

    Returns:
        result (`np.ndarray`): np array of decoded frames of shape (num_frames, height, width, 3).
    """
    frames = []
    container.seek(0)
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


# The following function is adapted from the Hugging Face Transformers documentation.
# Source: Transformers: VideoMAEModel Documentation
# URL: https://huggingface.co/docs/transformers/model_doc/videomae#transformers.VideoMAEModel.forward.example
def sample_frame_indices(num_frame_2_sample, frame_sample_rate, max_end_frame_idx):
    """
    Sample a given number of frame indices.

    Parameters:
        num_frame_2_sample (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        max_end_frame_idx (`int`): Maximum allowed index of sample's last frame.

    Returns:
        indices (`List[int]`): List of sampled frame indices.
    """
    total_num_frame = int(num_frame_2_sample * frame_sample_rate)
    end_idx = np.random.randint(total_num_frame, max_end_frame_idx)
    start_idx = end_idx - total_num_frame
    indices = np.linspace(start_idx, end_idx, num=num_frame_2_sample)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


def process_video(file_path, ckpt, num_frame_2_sample=16, frame_sample_rate=1):
    """
    Process a video file and return an input dict for VideoMAEModel.

    Parameters:
        file_path (`str`): Path to the video file.
        ckpt (`str`): The checkpoint of VideoMAEImageProcessor to use.
        frame_sample_rate (`int`): Sample every n-th frame.

    Returns:
        input torch.Tensor: Processed video for VideoMAEModel, which is a tensor of shape
        (num_frame_2_sample, 3, 224, 224)
    """
    container = av.open(file_path)

    # Get `num_frame_2_sample` frame indices
    indices = sample_frame_indices(num_frame_2_sample=num_frame_2_sample, frame_sample_rate=frame_sample_rate,
                                   max_end_frame_idx=container.streams.video[0].frames)

    # Get frames of given indices
    frames = get_frames(container, indices)

    # Process frames and return the input dict
    image_processor = VideoMAEImageProcessor.from_pretrained(ckpt)
    video = image_processor(list(frames), return_tensors="pt")['pixel_values'][0]
    return video


if __name__ == '__main__':
    import torch
    import glob

    ckpt = "MCG-NJU/videomae-base"
    num_frame_2_sample = 16
    frame_sample_rate = 2
    video_paths = glob.glob('../data/dev/Movie Clips/*.avi')
    inputs = [process_video(video_path, ckpt, num_frame_2_sample, frame_sample_rate) for video_path in video_paths]
    batched_inputs = torch.stack(inputs, dim=0)
    assert batched_inputs.shape == (len(video_paths), num_frame_2_sample, 3, 224, 224)
