
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random
import csv
import math

ROOT = Path(
    "dataset/marathi/2_New_Manually_Annotated/"
    "2_part_2/marathi"
)

GT_FILE = ROOT / "gt.txt"
IMAGE_DIR = ROOT / "images"

REPORT_DIR = Path("dataset/marathi/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Reproducible random sample
random.seed(42)

na_entries = []

for line in GT_FILE.read_text(encoding="utf-8-sig").splitlines():
    parts = line.split("\t", 1)

    if len(parts) == 2 and parts[1].strip().upper() == "NA":
        image_name = Path(parts[0]).name
        image_path = IMAGE_DIR / image_name

        if image_path.is_file():
            na_entries.append(image_name)

selected = random.sample(
    na_entries,
    min(100, len(na_entries))
)

# Contact sheet settings
cols = 4
cell_w = 300
cell_h = 130
rows = math.ceil(len(selected) / cols)

sheet = Image.new(
    "RGB",
    (cols * cell_w, rows * cell_h),
    "white"
)

draw = ImageDraw.Draw(sheet)

font_path = Path(r"C:\Windows\Fonts\Nirmala.ttc")

try:
    font = ImageFont.truetype(str(font_path), 18, index=0)
except Exception:
    font = ImageFont.load_default()

review_rows = []

for i, filename in enumerate(selected):

    image_path = IMAGE_DIR / filename

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        original_size = img.size
        img.thumbnail((275, 85))

    x = (i % cols) * cell_w
    y = (i // cols) * cell_h

    sheet.paste(img, (x + 8, y + 5))

    draw.text(
        (x + 8, y + 93),
        filename,
        fill="black",
        font=font
    )

    draw.text(
        (x + 8, y + 113),
        f"{original_size[0]} x {original_size[1]}",
        fill="gray"
    )

    review_rows.append({
        "image_name": filename,
        "original_transcript": "NA",
        "corrected_transcript": "",
        "readability": "",
        "review_notes": ""
    })

# Save contact sheet
sheet_path = REPORT_DIR / "na_review_100.png"
sheet.save(sheet_path)

# Save annotation review CSV
csv_path = REPORT_DIR / "na_review_100.csv"

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8-sig"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "image_name",
            "original_transcript",
            "corrected_transcript",
            "readability",
            "review_notes"
        ]
    )

    writer.writeheader()
    writer.writerows(review_rows)

print("NA images available:", len(na_entries))
print("Images selected:", len(selected))
print("Contact sheet:", sheet_path.resolve())
print("Review CSV:", csv_path.resolve())
