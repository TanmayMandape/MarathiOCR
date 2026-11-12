

from pathlib import Path

from htr_dataset import HTRDataset
from prepare_tensors import image_to_tensor
from vocabulary import build_vocabulary


# Project directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = PROJECT_ROOT / "dataset" / "marathi"
ANNOTATION_FILE = DATASET_ROOT / "file" / "train_gt.txt"


# Build character vocabulary from training labels
vocab = build_vocabulary()


# Create dataset using our existing functions
dataset = HTRDataset(
    annotation_file=ANNOTATION_FILE,
    dataset_root=DATASET_ROOT,
    image_fn=image_to_tensor,
    encode_fn=vocab.encode,
    max_samples=5,
    seed=42
)


print("\nDataset integration test")
print("=" * 40)

print("Number of samples:", len(dataset))


# Inspect each sample
for index in range(len(dataset)):

    sample = dataset[index]

    print(f"\nSample {index + 1}")
    print("Text:", sample["text"])
    print("Image shape:", sample["image"].shape)
    print("Image width:", sample["image_width"])
    print("Target:", sample["target"].tolist())
    print("Target length:", sample["target_length"])
    print("Image path:", sample["path"])


print("\nDataset integration test completed.")

