
from pathlib import Path
from collections import Counter
from PIL import Image
import json
import csv
import hashlib

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATASET_ROOT = Path("dataset/marathi")
ANNOTATION_DIR = DATASET_ROOT / "file"

REPORT_DIR = DATASET_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "val", "test"]

# --------------------------------------------------
# STORAGE FOR AUDIT RESULTS
# --------------------------------------------------

report = {
    "dataset": "Marathi Handwritten Dataset",
    "splits": {},
    "overall": {
        "total_annotations": 0,
        "total_valid_images": 0,
        "total_invalid_images": 0,
        "total_empty_transcripts": 0,
        "total_malformed_lines": 0,
        "duplicate_paths_within_splits": 0,
        "duplicate_paths_across_splits": 0
    }
}

all_paths = {}
problem_samples = []

# --------------------------------------------------
# ANNOTATION PARSER
# --------------------------------------------------

def parse_annotation(line):
    """
    Expected format:
    image_path<TAB>Marathi transcription
    """

    parts = line.split("\t", 1)

    if len(parts) != 2:
        parts = line.split(maxsplit=1)

    if len(parts) != 2:
        return None, None

    image_path = parts[0].strip()
    transcript = parts[1].strip()

    if not image_path:
        return None, None

    return image_path, transcript


# --------------------------------------------------
# AUDIT EACH SPLIT
# --------------------------------------------------

for split in SPLITS:

    print(f"\nAuditing {split.upper()} split...")

    annotation_file = ANNOTATION_DIR / f"{split}_gt.txt"

    if not annotation_file.exists():
        print(f"ERROR: Missing {annotation_file}")
        continue

    lines = annotation_file.read_text(
        encoding="utf-8-sig"
    ).splitlines()

    stats = {
        "annotation_lines": len(lines),
        "valid_images": 0,
        "invalid_images": 0,
        "empty_transcripts": 0,
        "malformed_lines": 0,
        "duplicate_paths": 0,
        "unique_image_paths": 0,
        "total_characters": 0,
        "total_words": 0,
        "image_modes": Counter(),
        "image_extensions": Counter(),
        "aspect_ratios": [],
        "widths": [],
        "heights": [],
        "image_sizes": Counter()
    }

    seen_paths = set()
    split_paths = set()

    for index, line in enumerate(lines, start=1):

        image_path_text, transcript = parse_annotation(line)

        if image_path_text is None:
            stats["malformed_lines"] += 1

            problem_samples.append({
                "split": split,
                "line": index,
                "image_path": "",
                "problem": "Malformed annotation line"
            })
            continue

        image_path = DATASET_ROOT / Path(
            image_path_text.replace("/", "\\")
        )

        normalized_path = image_path_text.replace("\\", "/").lower()

        # Duplicate detection within split
        if normalized_path in seen_paths:
            stats["duplicate_paths"] += 1

            problem_samples.append({
                "split": split,
                "line": index,
                "image_path": image_path_text,
                "problem": "Duplicate path within split"
            })

        seen_paths.add(normalized_path)
        split_paths.add(normalized_path)

        # Empty transcript check
        if not transcript:
            stats["empty_transcripts"] += 1

            problem_samples.append({
                "split": split,
                "line": index,
                "image_path": image_path_text,
                "problem": "Empty transcript"
            })

        else:
            stats["total_characters"] += len(transcript)
            stats["total_words"] += len(transcript.split())

        # Image existence check
        if not image_path.is_file():

            stats["invalid_images"] += 1

            problem_samples.append({
                "split": split,
                "line": index,
                "image_path": image_path_text,
                "problem": "Image file missing"
            })

            continue

        # Image integrity check
        try:
            with Image.open(image_path) as img:
                img.verify()

            with Image.open(image_path) as img:

                width, height = img.size

                stats["valid_images"] += 1
                stats["widths"].append(width)
                stats["heights"].append(height)

                stats["image_modes"][img.mode] += 1
                stats["image_extensions"][image_path.suffix.lower()] += 1
                stats["image_sizes"][f"{width}x{height}"] += 1

                if height > 0:
                    stats["aspect_ratios"].append(
                        round(width / height, 3)
                    )

        except Exception as error:

            stats["invalid_images"] += 1

            problem_samples.append({
                "split": split,
                "line": index,
                "image_path": image_path_text,
                "problem": f"Unreadable image: {error}"
            })

    # Cross-split overlap detection
    for previous_split, previous_paths in all_paths.items():

        overlap = split_paths.intersection(previous_paths)

        if overlap:
            report["overall"][
                "duplicate_paths_across_splits"
            ] += len(overlap)

            for path in list(overlap)[:20]:
                problem_samples.append({
                    "split": split,
                    "line": "",
                    "image_path": path,
                    "problem": f"Also appears in {previous_split}"
                })

    all_paths[split] = split_paths

    stats["unique_image_paths"] = len(split_paths)

    # Convert counters into JSON-compatible dictionaries
    stats["image_modes"] = dict(stats["image_modes"])
    stats["image_extensions"] = dict(stats["image_extensions"])
    stats["image_sizes"] = dict(
        stats["image_sizes"].most_common(20)
    )

    # Dimension statistics
    if stats["widths"]:

        stats["dimensions"] = {
            "min_width": min(stats["widths"]),
            "max_width": max(stats["widths"]),
            "average_width": round(
                sum(stats["widths"]) / len(stats["widths"]), 2
            ),
            "min_height": min(stats["heights"]),
            "max_height": max(stats["heights"]),
            "average_height": round(
                sum(stats["heights"]) / len(stats["heights"]), 2
            )
        }

        stats["aspect_ratio_statistics"] = {
            "min": min(stats["aspect_ratios"]),
            "max": max(stats["aspect_ratios"]),
            "average": round(
                sum(stats["aspect_ratios"]) /
                len(stats["aspect_ratios"]), 3
            )
        }

    # Remove large internal arrays before saving
    for key in [
        "widths",
        "heights",
        "aspect_ratios"
    ]:
        stats.pop(key, None)

    # Update overall report
    report["splits"][split] = stats

    report["overall"]["total_annotations"] += stats["annotation_lines"]
    report["overall"]["total_valid_images"] += stats["valid_images"]
    report["overall"]["total_invalid_images"] += stats["invalid_images"]
    report["overall"]["total_empty_transcripts"] += stats["empty_transcripts"]
    report["overall"]["total_malformed_lines"] += stats["malformed_lines"]
    report["overall"]["duplicate_paths_within_splits"] += stats["duplicate_paths"]

    print(f"Annotations: {stats['annotation_lines']}")
    print(f"Valid images: {stats['valid_images']}")
    print(f"Invalid images: {stats['invalid_images']}")
    print(f"Empty transcripts: {stats['empty_transcripts']}")
    print(f"Duplicate paths: {stats['duplicate_paths']}")

# --------------------------------------------------
# SAVE JSON REPORT
# --------------------------------------------------

json_path = REPORT_DIR / "marathi_dataset_audit.json"

with open(json_path, "w", encoding="utf-8") as file:
    json.dump(report, file, ensure_ascii=False, indent=4)

# --------------------------------------------------
# SAVE PROBLEM SAMPLES CSV
# --------------------------------------------------

csv_path = REPORT_DIR / "marathi_dataset_problems.csv"

with open(csv_path, "w", newline="", encoding="utf-8-sig") as file:

    if problem_samples:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "split",
                "line",
                "image_path",
                "problem"
            ]
        )

        writer.writeheader()
        writer.writerows(problem_samples)

    else:
        writer = csv.writer(file)
        writer.writerow([
            "split",
            "line",
            "image_path",
            "problem"
        ])

# --------------------------------------------------
# FINAL SUMMARY
# --------------------------------------------------

print("\n" + "=" * 45)
print("MARATHI DATASET AUDIT COMPLETED")
print("=" * 45)

for key, value in report["overall"].items():
    print(f"{key}: {value}")

print("\nReports saved:")
print(json_path.resolve())
print(csv_path.resolve())

