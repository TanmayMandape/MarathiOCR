
import pandas as pd
from PIL import Image
from io import BytesIO
from pathlib import Path

# Locate the dataset
project_dir = Path(__file__).resolve().parent.parent
dataset_path = project_dir / "dataset" / "train-00000-of-00001.parquet"

# Read the Parquet file
df = pd.read_parquet(dataset_path)

# Create an output folder
output_dir = project_dir / "dataset" / "samples"
output_dir.mkdir(parents=True, exist_ok=True)

# Extract the first five images
for index, (_, row) in enumerate(df.head(5).iterrows()):
    image_bytes = row["image"]["bytes"]
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    output_path = output_dir / f"sample_{index}.png"
    image.save(output_path)

    print(f"Sample {index + 1}")
    print(f"  Label: {row['label']}")
    print(f"  Type: {row['type']}")
    print(f"  Image: {output_path}")

print(f"\nExtracted 5 images to: {output_dir}")
