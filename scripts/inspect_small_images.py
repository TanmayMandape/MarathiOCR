
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math

root = Path("dataset/marathi/2_New_Manually_Annotated/2_part_2/marathi")
images_dir = root / "images"
gt_file = root / "gt.txt"

# Load image-to-transcript mapping
annotations = {}

for line in gt_file.read_text(encoding="utf-8-sig").splitlines():
    parts = line.split("\t", 1)
    if len(parts) == 2:
        annotations[Path(parts[0]).name] = parts[1].strip()

# Find images with either dimension below 32 pixels
selected = []

for path in images_dir.glob("*.jpg"):
    with Image.open(path) as img:
        if img.width < 32 or img.height < 32:
            selected.append(path)

selected.sort()

# Create contact sheet
cols = 4
cell_w, cell_h = 300, 125
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

for i, path in enumerate(selected):
    with Image.open(path) as img:
        img = img.convert("RGB")

        original_size = img.size

        # Enlarge for visual inspection only
        img.thumbnail((270, 65))

        scale = min(270 / original_size[0], 65 / original_size[1])
        display = Image.open(path).convert("RGB").resize(
            (
                max(1, int(original_size[0] * scale)),
                max(1, int(original_size[1] * scale))
            )
        )

    x = (i % cols) * cell_w
    y = (i // cols) * cell_h

    sheet.paste(display, (x + 8, y + 5))

    transcript = annotations.get(path.name, "NO ANNOTATION")

    draw.text(
        (x + 8, y + 77),
        transcript[:30],
        fill="black",
        font=font
    )

    draw.text(
        (x + 8, y + 103),
        f"{path.name} | {original_size}",
        fill="gray"
    )

output = Path("dataset/marathi/reports")
output.mkdir(parents=True, exist_ok=True)

destination = output / "small_images_contact_sheet.png"
sheet.save(destination)

print("Images inspected:", len(selected))
print("Saved:", destination.resolve())
