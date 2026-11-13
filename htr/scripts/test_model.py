
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add project root and scripts directory to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "htr" / "scripts"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from htr.models.crnn import CRNN
from htr.scripts.htr_dataset import HTRDataset
from htr.scripts.collate import collate_fn
from prepare_tensors import image_to_tensor
from vocabulary import build_vocabulary


def test_real_batch():

    print("=" * 55)
    print("MARATHIOCR - REAL DATA MODEL TEST")
    print("=" * 55)

    # 1. Build vocabulary from training annotations
    vocabulary = build_vocabulary()

    # 2. Locate training annotations
    dataset_root = PROJECT_ROOT / "dataset" / "marathi"
    annotation_file = dataset_root / "file" / "train_gt.txt"

    # 3. Create dataset using actual images and labels
    dataset = HTRDataset(
        annotation_file=annotation_file,
        dataset_root=dataset_root,
        image_fn=image_to_tensor,
        encode_fn=vocabulary.encode,
        max_samples=8,
        seed=100
    )

    print("\nDataset samples:", len(dataset))

    # 4. Create a batch of 8 real images
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn
    )

    batch = next(iter(loader))

    images = batch["images"]
    image_widths = batch["image_widths"]
    targets = batch["targets"]
    target_lengths = batch["target_lengths"]

    print("\nREAL INPUT BATCH")
    print("Images:", images.shape)
    print("Image widths:", image_widths.tolist())
    print("Targets:", targets.shape)
    print("Target lengths:", target_lengths.tolist())

    # 5. Pass real images through CRNN
    device = torch.device("cpu")

    model = CRNN(
        num_classes=vocabulary.vocab_size,
        hidden_size=128
    ).to(device)

    model.eval()

    with torch.no_grad():
        predictions = model(images.to(device))

    print("\nMODEL OUTPUT")
    print("Output shape:", predictions.shape)

    # 6. Calculate valid output lengths for CTC
    output_lengths = image_widths // 4

    print("\nCTC SEQUENCE LENGTHS")
    print("Input widths:", image_widths.tolist())
    print("Output lengths:", output_lengths.tolist())
    print("Model output time steps:", predictions.shape[1])

    # 7. Validate results
    assert images.shape[0] == 8
    assert images.shape[1] == 1
    assert images.shape[2] == 64
    assert predictions.shape[0] == 8
    assert predictions.shape[2] == vocabulary.vocab_size
    assert torch.all(output_lengths <= predictions.shape[1])
    assert torch.all(target_lengths > 0)

    print("\nACTUAL TRANSCRIPTIONS")
    for i, text in enumerate(batch["texts"]):
        print(f"{i + 1}. {text}")

    print("\nALL REAL DATA MODEL TESTS PASSED")


if __name__ == "__main__":
    test_real_batch()
