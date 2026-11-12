
from pathlib import Path
from PIL import Image
from collections import Counter
import csv
import json

ROOT = Path("dataset/marathi/2_New_Manually_Annotated")
PARTS = ["1_part_1", "2_part_2"]

results = {}

for part in PARTS:
    folder = ROOT / part / "marathi"
    annotation_file = folder / "gt.txt"

    lines = annotation_file.read_text(
        encoding="utf-8-sig"
    ).splitlines()

    valid = 0
    invalid = 0
    empty = 0
    malformed = 0
    widths = []
    heights = []
    modes = Counter()
    issues = []

    seen = set()

    for index, line in enumerate(lines, 1):
        fields = line.split("\t", 1)

        if len(fields) != 2:
            malformed += 1
            issues.append({
                "line": index,
                "problem": "Malformed annotation"
            })
            continue

        relative_path, transcript = fields
        image_path = folder / relative_path

        if not transcript.strip():
            empty += 1
            issues.append({
                "line": index,
                "problem": "Empty transcript"
            })

        if relative_path in seen:
            issues.append({
                "line": index,
                "problem": "Duplicate image path"
            })

        seen.add(relative_path)

        if not image_path.is_file():
            invalid += 1
            issues.append({
                "line": index,
                "problem": "Missing image"
            })
            continue

        try:
            with Image.open(image_path) as img:
                img.verify()

            with Image.open(image_path) as img:
                valid += 1
                widths.append(img.width)
                heights.append(img.height)
                modes[img.mode] += 1

        except Exception as error:
            invalid += 1
            issues.append({
                "line": index,
                "problem": f"Unreadable image: {error}"
            })

    results[part] = {
        "annotation_lines": len(lines),
        "valid_images": valid,
        "invalid_images": invalid,
        "empty_transcripts": empty,
        "malformed_lines": malformed,
        "unique_paths": len(seen),
        "duplicate_paths": len(lines) - len(seen),
        "image_modes": dict(modes),
        "dimensions": {
            "min_width": min(widths) if widths else None,
            "max_width": max(widths) if widths else None,
            "min_height": min(heights) if heights else None,
            "max_height": max(heights) if heights else None
        }
    }

    report_dir = ROOT / "reports"
    report_dir.mkdir(exist_ok=True)

    with open(
        report_dir / f"{part}_issues.csv",
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["line", "problem"]
        )
        writer.writeheader()
        writer.writerows(issues)

    print(f"\n{part}")
    print(json.dumps(results[part], indent=4, ensure_ascii=False))

report_dir = ROOT / "reports"
with open(
    report_dir / "manual_dataset_audit.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print("\nManual dataset audit completed.")
