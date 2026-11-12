
from pathlib import Path

from dataset_loader import load_annotations
from prepare_tensors import image_to_tensor
from vocabulary import build_vocabulary
from htr_dataset import HTRDataset
from collate import collate_fn


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "dataset" / "marathi"
ANNOTATION_FILE = DATASET_ROOT / "file" / "train_gt.txt"


# Build vocabulary from training annotations
vocabulary = build_vocabulary()


# Load 8 reproducible training samples
dataset = HTRDataset(
    annotation_file=ANNOTATION_FILE,
    dataset_root=DATASET_ROOT,
    image_fn=image_to_tensor,
    encode_fn=vocabulary.encode,
    max_samples=8,
    seed=42,
)


# Prepare one batch containing all 8 samples
samples = [dataset[i] for i in range(8)]
batch = collate_fn(samples)


# Display results
print("\nCOLLATE FUNCTION TEST")
print("=" * 45)

print("Number of images:", batch["images"].shape[0])
print("Batch tensor shape:", batch["images"].shape)
print("Image widths:", batch["image_widths"].tolist())

print("Combined target shape:", batch["targets"].shape)
print("Target lengths:", batch["target_lengths"].tolist())

print("\nRecognized training labels:")
for text in batch["texts"]:
    print("-", text)


# Basic validation
assert batch["images"].shape[0] == 8
assert batch["images"].shape[1] == 1
assert batch["images"].shape[2] == 64
assert len(batch["texts"]) == 8
assert len(batch["paths"]) == 8
assert batch["targets"].numel() == batch["target_lengths"].sum().item()

print("\nAll collate tests passed.")
