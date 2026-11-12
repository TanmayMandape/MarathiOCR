
import random
from PIL import Image
from dataset_loader import load_annotations

random.seed(42)

samples = load_annotations("train")
sample_size = min(1000, len(samples))
selected = random.sample(samples, sample_size)

widths = []
heights = []
failed = 0

for sample in selected:
    try:
        with Image.open(sample["image_path"]) as image:
            width, height = image.size
            widths.append(width)
            heights.append(height)
    except Exception:
        failed += 1

print("Sample size:", sample_size)
print("Successfully inspected:", len(widths))
print("Failed:", failed)

print("\nWidth:")
print("Minimum:", min(widths))
print("Maximum:", max(widths))
print("Average:", round(sum(widths) / len(widths), 2))

print("\nHeight:")
print("Minimum:", min(heights))
print("Maximum:", max(heights))
print("Average:", round(sum(heights) / len(heights), 2))
