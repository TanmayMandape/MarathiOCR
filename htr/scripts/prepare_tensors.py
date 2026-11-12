
import torch
import numpy as np

from preprocess import preprocess_image
from dataset_loader import load_annotations


def image_to_tensor(image_path):
    """
    Convert a preprocessed grayscale image
    into a PyTorch tensor.
    """

    image = preprocess_image(image_path)

    # Convert image into a NumPy array
    image_array = np.array(image, dtype=np.float32)

    # Normalize pixel values from 0-255 to 0-1
    image_array = image_array / 255.0

    # Convert NumPy array into a PyTorch tensor
    tensor = torch.from_numpy(image_array)

    # Add channel dimension: (H, W) -> (1, H, W)
    tensor = tensor.unsqueeze(0)

    return tensor


if __name__ == "__main__":

    samples = load_annotations("train")

    for sample in samples[:5]:

        tensor = image_to_tensor(sample["image_path"])

        print("Text:", sample["text"])
        print("Tensor shape:", tensor.shape)
        print("Data type:", tensor.dtype)
        print("Minimum pixel:", tensor.min().item())
        print("Maximum pixel:", tensor.max().item())
        print("-" * 40)
