
from PIL import Image, ImageOps


TARGET_HEIGHT = 64
PADDING_VALUE = 255


def preprocess_image(image_path):
    """
    Preprocess a handwritten Marathi word image.

    Steps:
    1. Convert to grayscale.
    2. Resize to 64 pixels in height.
    3. Preserve aspect ratio.
    4. Pad width to a multiple of 32.
    """

    image = Image.open(image_path).convert("L")

    original_width, original_height = image.size

    # Calculate new width while preserving aspect ratio
    new_width = round(
        original_width * TARGET_HEIGHT / original_height
    )

    # Resize image
    image = image.resize(
        (new_width, TARGET_HEIGHT),
        Image.Resampling.LANCZOS
    )

    # Pad width to the next multiple of 32
    padded_width = ((new_width + 31) // 32) * 32

    image = ImageOps.pad(
        image,
        (padded_width, TARGET_HEIGHT),
        method=Image.Resampling.LANCZOS,
        color=PADDING_VALUE,
        centering=(0, 0)
    )

    return image


if __name__ == "__main__":

    from dataset_loader import load_annotations

    samples = load_annotations("train")

    for sample in samples[:5]:

        processed = preprocess_image(sample["image_path"])

        print("Text:", sample["text"])
        print("Processed dimensions:", processed.size)
        print("-" * 40)

    # Save one example for visual inspection
    output_path = (
        "htr/reports/preprocessed_sample.png"
    )

    processed.save(output_path)

    print("Sample saved:", output_path)
