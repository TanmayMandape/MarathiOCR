
from pathlib import Path
from io import BytesIO
from collections import Counter

import pandas as pd
from PIL import Image, UnidentifiedImageError


# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset"

FILES = {
    "train": DATASET_DIR / "train-00000-of-00001.parquet",
    "validation": DATASET_DIR / "validation-00000-of-00001.parquet",
}

REPORT_DIR = DATASET_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Validate one dataset split
# --------------------------------------------------

def validate_split(split_name, file_path):
    print(f"\n{'=' * 60}")
    print(f"VALIDATING: {split_name.upper()}")
    print(f"{'=' * 60}")

    if not file_path.exists():
        print(f"ERROR: File not found: {file_path}")
        return None

    df = pd.read_parquet(file_path)

    required_columns = {"image", "id", "filename", "label", "type"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        print(f"ERROR: Missing columns: {sorted(missing_columns)}")
        return None

    print(f"Rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")

    # Check metadata
    missing_labels = df["label"].isna() | (
        df["label"].astype(str).str.strip() == ""
    )
    missing_images = df["image"].isna()
    missing_filenames = df["filename"].isna()

    duplicate_ids = int(df["id"].duplicated().sum())
    duplicate_filenames = int(df["filename"].duplicated().sum())

    type_counts = Counter(
        df["type"].fillna("MISSING").astype(str)
    )

    print("\nMetadata checks")
    print(f"Missing labels: {int(missing_labels.sum())}")
    print(f"Missing image fields: {int(missing_images.sum())}")
    print(f"Missing filenames: {int(missing_filenames.sum())}")
    print(f"Duplicate IDs: {duplicate_ids}")
    print(f"Duplicate filenames: {duplicate_filenames}")
    print(f"Categories: {dict(type_counts)}")

    # Inspect image bytes and dimensions
    image_errors = []
    image_sizes = Counter()
    image_modes = Counter()
    invalid_image_rows = []

    for index, row in df.iterrows():
        image_data = row["image"]

        if not isinstance(image_data, dict):
            image_errors.append({
                "row": index,
                "filename": row["filename"],
                "error": "Image field is not a dictionary",
            })
            invalid_image_rows.append(index)
            continue

        image_bytes = image_data.get("bytes")

        if not image_bytes:
            image_errors.append({
                "row": index,
                "filename": row["filename"],
                "error": "Missing image bytes",
            })
            invalid_image_rows.append(index)
            continue

        try:
            # Verify that the bytes represent a readable image.
            with Image.open(BytesIO(image_bytes)) as image:
                image.verify()

            # Reopen after verify() to read image metadata.
            with Image.open(BytesIO(image_bytes)) as image:
                image_sizes[image.size] += 1
                image_modes[image.mode] += 1

        except (UnidentifiedImageError, OSError, ValueError) as exc:
            image_errors.append({
                "row": index,
                "filename": row["filename"],
                "error": str(exc),
            })
            invalid_image_rows.append(index)

    print("\nImage checks")
    print(f"Images with errors: {len(image_errors)}")
    print(f"Image dimensions (width, height): {dict(image_sizes)}")
    print(f"Image modes: {dict(image_modes)}")

    # Save errors for later investigation.
    errors_path = REPORT_DIR / f"{split_name}_image_errors.csv"
    pd.DataFrame(
        image_errors,
        columns=["row", "filename", "error"]
    ).to_csv(errors_path, index=False, encoding="utf-8-sig")

    # Save a concise report.
    report = {
        "split": split_name,
        "file": str(file_path),
        "rows": len(df),
        "missing_labels": int(missing_labels.sum()),
        "missing_images": int(missing_images.sum()),
        "missing_filenames": int(missing_filenames.sum()),
        "duplicate_ids": duplicate_ids,
        "duplicate_filenames": duplicate_filenames,
        "invalid_images": len(image_errors),
        "categories": dict(type_counts),
        "image_dimensions": {
            f"{width}x{height}": count
            for (width, height), count in image_sizes.items()
        },
        "image_modes": dict(image_modes),
    }

    return report


# --------------------------------------------------
# 3. Run validation on both splits
# --------------------------------------------------

def main():
    reports = []

    for split_name, file_path in FILES.items():
        report = validate_split(split_name, file_path)

        if report is not None:
            reports.append(report)

    if not reports:
        print("\nNo reports generated. Check the file paths.")
        return

    report_path = REPORT_DIR / "dataset_validation_report.json"

    import json
    report_path.write_text(
        json.dumps(reports, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n{'=' * 60}")
    print("VALIDATION COMPLETE")
    print(f"Report saved to: {report_path}")
    print(f"Error CSVs saved in: {REPORT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
