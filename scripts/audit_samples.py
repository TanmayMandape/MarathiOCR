from pathlib import Path
import random

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset"
OUTPUT_DIR = DATASET_DIR / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)

# Use a Windows font that supports Devanagari.
font_candidates = [
    Path("C:/Windows/Fonts/Nirmala.ttc"),
]

font_path = next((p for p in font_candidates if p.exists()), None)

if font_path is None:
    raise FileNotFoundError("Nirmala Devanagari font not found.")

font = ImageFont.truetype(str(font_path), 18, index=0)

font_path = next((p for p in font_candidates if p.exists()), None)
font = ImageFont.truetype(str(font_path), 18) if font_path else ImageFont.load_default()

def make_contact_sheet(split_name, filename, samples_per_type=12):
    df = pd.read_parquet(DATASET_DIR / filename)

    selected = []
    for category in sorted(df["type"].dropna().unique()):
        group = df[df["type"] == category]
        count = min(samples_per_type, len(group))
        selected.append(group.sample(n=count, random_state=42))

    sample_df = pd.concat(selected).reset_index(drop=True)

    cell_w, cell_h = 280, 135
    columns = 3
    rows = (len(sample_df) + columns - 1) // columns

    sheet = Image.new(
        "RGB",
        (columns * cell_w, rows * cell_h),
        "white",
    )
    draw = ImageDraw.Draw(sheet)

    for i, row in sample_df.iterrows():
        image_bytes = row["image"]["bytes"]

        from io import BytesIO
        import_image = Image.open(BytesIO(image_bytes)).convert("RGBA")

        # Put transparent pixels on a white background.
        background = Image.new("RGBA", import_image.size, "white")
        background.alpha_composite(import_image)
        image = background.convert("RGB")

        # Enlarge images for visual inspection.
        image.thumbnail((cell_w - 20, 75))

        x = (i % columns) * cell_w
        y = (i // columns) * cell_h

        sheet.paste(image, (x + 10, y + 5))
        draw.text(
            (x + 10, y + 84),
            f"Label: {row['label']}",
            fill="black",
            font=font,
        )
        draw.text(
            (x + 10, y + 108),
            f"Type: {row['type']} | ID: {row['id']}",
            fill="black",
        )

    output_path = OUTPUT_DIR / f"{split_name}_contact_sheet.png"
    sheet.save(output_path)
    print(f"Saved: {output_path}")

make_contact_sheet(
    "train",
    "train-00000-of-00001.parquet",
)

make_contact_sheet(
    "validation",
    "validation-00000-of-00001.parquet",
)

print("\nOpen the two contact sheets and check whether the labels match.")
