
import torch


def collate_fn(batch):
    """
    Combines variable-width handwritten word images
    into one batch for PyTorch training.
    """

    # Find the maximum image width in this batch
    max_width = max(sample["image_width"] for sample in batch)

    images = []
    image_widths = []
    targets = []
    target_lengths = []
    texts = []
    paths = []

    for sample in batch:
        image = sample["image"]

        # Calculate how much white padding is required
        padding_width = max_width - image.shape[-1]

        # Add white padding to the right
        padded_image = torch.nn.functional.pad(
            image,
            (0, padding_width, 0, 0),
            value=1.0
        )

        images.append(padded_image)

        # Preserve the original width before padding
        image_widths.append(sample["image_width"])

        targets.append(sample["target"])
        target_lengths.append(sample["target_length"])

        texts.append(sample["text"])
        paths.append(sample["path"])

    # Stack images into a single tensor
    image_batch = torch.stack(images)

    # Combine individual target sequences
    target_batch = torch.cat(targets)

    return {
        "images": image_batch,
        "image_widths": torch.tensor(image_widths, dtype=torch.long),
        "targets": target_batch,
        "target_lengths": torch.tensor(target_lengths, dtype=torch.long),
        "texts": texts,
        "paths": paths,
    }
