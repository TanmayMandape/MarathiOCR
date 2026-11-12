
from pathlib import Path
from PIL import Image


# Project directory: D:\MPSC
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = PROJECT_ROOT / "dataset" / "marathi"
ANNOTATION_DIR = DATASET_ROOT / "file"


def load_annotations(split="train"):
    """
    Load image paths and Marathi transcriptions
    from the selected dataset split.
    """

    annotation_file = ANNOTATION_DIR / f"{split}_gt.txt"

    if not annotation_file.exists():
        raise FileNotFoundError(
            f"Annotation file not found: {annotation_file}"
        )

    samples = []

    with open(annotation_file, "r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):

            line = line.rstrip("\n\r")

            if not line.strip():
                continue

            parts = line.split("\t", 1)

            if len(parts) != 2:
                print(f"Skipping malformed line: {line_number}")
                continue

            relative_path, transcription = parts

            image_path = DATASET_ROOT / relative_path

            samples.append({
                "image_path": image_path,
                "text": transcription
            })

    return samples


def load_image(image_path):
    """Open a handwritten word image as RGB."""

    with Image.open(image_path) as image:
        return image.convert("RGB")


if __name__ == "__main__":

    train_samples = load_annotations("train")

    print("Total training samples:", len(train_samples))

    print("\nFirst 5 samples:")

    for sample in train_samples[:5]:
        print("Text:", sample["text"])
        print("Image:", sample["image_path"].name)

        image = load_image(sample["image_path"])
        print("Image dimensions:", image.size)
        print("-" * 40)
