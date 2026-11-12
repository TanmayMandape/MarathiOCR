"""PyTorch Dataset for word-level Marathi handwritten text recognition.
 
Annotation file format (one sample per line):
    relative_image_path<TAB>Marathi_transcription
 
The class does not import your preprocessing or vocabulary code directly.
Instead you pass in two functions:
    image_fn(image_path)  -> torch.FloatTensor of shape (1, H, W)
    encode_fn(text)       -> list[int]   (character indices, blank = 0 is never used here)
This keeps the class testable and independent of how those functions are written.
"""
import random
from pathlib import Path
 
import torch
from torch.utils.data import Dataset
 
 
class HTRDataset(Dataset):
    def __init__(
        self,
        annotation_file,
        dataset_root,
        image_fn,
        encode_fn,
        max_samples=None,
        seed=0,
    ):
        self.root = Path(dataset_root)
        self.image_fn = image_fn
        self.encode_fn = encode_fn
        self.samples = self._read_annotations(Path(annotation_file))
 
        # Optional reproducible subset, for fast pipeline checks.
        if max_samples is not None and max_samples < len(self.samples):
            rng = random.Random(seed)
            self.samples = rng.sample(self.samples, max_samples)
 
    @staticmethod
    def _read_annotations(path):
        samples = []
        # utf-8 is required: Windows would otherwise default to cp1252 and corrupt Devanagari.
        with path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.rstrip("\r\n")
                if not line:
                    continue
                parts = line.split("\t", maxsplit=1)
                if len(parts) != 2:
                    raise ValueError(f"{path}:{line_no} is not 'path<TAB>text': {line!r}")
                samples.append((parts[0], parts[1]))
        return samples
 
    def __len__(self):
        return len(self.samples)
 
    def __getitem__(self, index):
        rel_path, text = self.samples[index]
        image_path = self.root / rel_path
 
        image = self.image_fn(image_path)            # (1, H, W) float32
        target = torch.tensor(self.encode_fn(text), dtype=torch.long)
 
        return {
            "image": image,
            "image_width": image.shape[-1],          # real width before batch padding
            "target": target,
            "target_length": len(target),
            "text": text,                            # kept for debugging and evaluation
            "path": rel_path,
        }
