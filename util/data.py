import numpy as np


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
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
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
