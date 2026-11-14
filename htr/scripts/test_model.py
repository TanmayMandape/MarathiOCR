
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from PIL import Image

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


def validate_ctc_feasibility(annotation_file, dataset_root, vocabulary):

    print("\n" + "=" * 55)
    print("CTC FEASIBILITY VALIDATION")
    print("=" * 55)

    total_samples = 0
    feasible_samples = 0
    infeasible_samples = 0

    min_margin = float("inf")
    worst_samples = []

    with open(annotation_file, "r", encoding="utf-8-sig") as file:

        for line_number, line in enumerate(file, start=1):

            line = line.rstrip("\r\n")

            if not line:
                continue

            parts = line.split("\t", maxsplit=1)

            if len(parts) != 2:
                raise ValueError(f"Invalid annotation at line {line_number}")

            relative_path, text = parts
            image_path = dataset_root / relative_path

            # Read image dimensions without loading image pixels
            with Image.open(image_path) as image:
                original_width, original_height = image.size

            # Reproduce the width calculation from preprocess.py
            resized_width = round(
                original_width * 64 / original_height
            )

            padded_width = ((resized_width + 31) // 32) * 32

            # CNN reduces width by a factor of 4
            input_length = padded_width // 4

            # Encode transcription
            target = vocabulary.encode(text)

            target_length = len(target)

            # Count adjacent repeated character IDs
            repeated_count = sum(
                target[i] == target[i - 1]
                for i in range(1, target_length)
            )

            # Minimum CTC time steps required
            minimum_required = target_length + repeated_count

            margin = input_length - minimum_required

            total_samples += 1
            min_margin = min(min_margin, margin)

            if margin >= 0:
                feasible_samples += 1
            else:
                infeasible_samples += 1

                if len(worst_samples) < 10:
                    worst_samples.append({
                        "line": line_number,
                        "text": text,
                        "input_length": input_length,
                        "required": minimum_required,
                        "margin": margin
                    })

    print("\nTOTAL SAMPLES:", total_samples)
    print("CTC FEASIBLE:", feasible_samples)
    print("CTC INFEASIBLE:", infeasible_samples)
    print("Minimum sequence margin:", min_margin)

    if worst_samples:
        print("\nINFEASIBLE SAMPLE EXAMPLES:")

        for sample in worst_samples:
            print(sample)

    print("\nCTC FEASIBILITY TEST:",
          "PASSED" if infeasible_samples == 0 else "FAILED")

    return infeasible_samples == 0

vocabulary = build_vocabulary()
dataset_root = PROJECT_ROOT / "dataset" / "marathi"
annotation_file = dataset_root / "file" / "train_gt.txt"

ctc_feasible = validate_ctc_feasibility(
    annotation_file,
    dataset_root,
    vocabulary
)

if not ctc_feasible:
    raise RuntimeError(
        "Some training samples are not feasible for CTC."
    )