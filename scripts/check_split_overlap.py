
from pathlib import Path
from collections import defaultdict

ROOT = Path("dataset/marathi")
ANNOTATION_DIR = ROOT / "file"

SPLITS = ["train", "val", "test"]

group_sets = {}
image_sets = {}

for split in SPLITS:

    annotation_file = ANNOTATION_DIR / f"{split}_gt.txt"

    groups = set()
    images = set()

    lines = annotation_file.read_text(
        encoding="utf-8-sig"
    ).splitlines()

    for line in lines:
        parts = line.split("\t", 1)

        if len(parts) != 2:
            continue

        image_path = Path(parts[0].replace("\\", "/"))

        # Expected structure:
        # writer_or_group/image/filename.jpg
        if len(image_path.parts) >= 2:
            groups.add(image_path.parts[-3])

        images.add(image_path.name.lower())

    group_sets[split] = groups
    image_sets[split] = images

    print(f"\n{split.upper()}")
    print("Unique groups:", len(groups))
    print("Unique image filenames:", len(images))

print("\n" + "=" * 45)
print("GROUP OVERLAP ANALYSIS")
print("=" * 45)

for i, split_a in enumerate(SPLITS):
    for split_b in SPLITS[i + 1:]:

        overlap = group_sets[split_a] & group_sets[split_b]

        print(f"\n{split_a} vs {split_b}")
        print("Overlapping groups:", len(overlap))
        print("Examples:", sorted(overlap)[:10])

print("\n" + "=" * 45)
print("IMAGE FILENAME OVERLAP")
print("=" * 45)

for i, split_a in enumerate(SPLITS):
    for split_b in SPLITS[i + 1:]:

        overlap = image_sets[split_a] & image_sets[split_b]

        print(f"\n{split_a} vs {split_b}")
        print("Overlapping filenames:", len(overlap))
        print("Examples:", sorted(overlap)[:10])
