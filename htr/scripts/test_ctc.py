
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "htr" / "scripts"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from htr.models.crnn import CRNN
from htr.scripts.htr_dataset import HTRDataset
from htr.scripts.collate import collate_fn
from prepare_tensors import image_to_tensor
from vocabulary import build_vocabulary


def test_ctc():

    print("=" * 55)
    print("MARATHIOCR - CTC LOSS TEST")
    print("=" * 55)

    # 1. Build vocabulary
    vocabulary = build_vocabulary()

    # 2. Dataset paths
    dataset_root = PROJECT_ROOT / "dataset" / "marathi"
    annotation_file = dataset_root / "file" / "train_gt.txt"

    # 3. Load a reproducible batch of real samples
    dataset = HTRDataset(
        annotation_file=annotation_file,
        dataset_root=dataset_root,
        image_fn=image_to_tensor,
        encode_fn=vocabulary.encode,
        max_samples=8,
        seed=100
    )

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn
    )

    batch = next(iter(loader))

    images = batch["images"]
    targets = batch["targets"]
    target_lengths = batch["target_lengths"]
    image_widths = batch["image_widths"]

    print("\nREAL DATA")
    print("Images:", images.shape)
    print("Targets:", targets.shape)
    print("Target lengths:", target_lengths.tolist())

    # 4. Calculate valid output sequence lengths
    output_lengths = image_widths // 4

    print("Output lengths:", output_lengths.tolist())

    # 5. Create CRNN
    device = torch.device("cpu")

    model = CRNN(
        num_classes=vocabulary.vocab_size,
        hidden_size=128
    ).to(device)

    model.eval()

    # 6. Forward pass
    predictions = model(images.to(device))

    print("\nMODEL PREDICTIONS")
    print("Shape:", predictions.shape)

    # 7. Convert raw scores into log probabilities
    log_probs = F.log_softmax(predictions, dim=2)

    # CTC expects (Time, Batch, Classes)
    log_probs = log_probs.permute(1, 0, 2)

    print("CTC input shape:", log_probs.shape)

    # 8. Calculate CTC loss
    ctc_loss_fn = torch.nn.CTCLoss(
        blank=0,
        reduction="mean",
        zero_infinity=False
    )

    loss = ctc_loss_fn(
        log_probs,
        targets,
        output_lengths,
        target_lengths
    )

    print("\nCTC LOSS")
    print("Loss:", loss.item())
    print("Finite:", torch.isfinite(loss).item())

    assert torch.isfinite(loss), "CTC loss is not finite"

    # 9. Backpropagation test
    model.zero_grad()
    loss.backward()

    gradients_exist = all(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    assert gradients_exist, "Missing gradients"

    gradients_finite = all(
        torch.isfinite(parameter.grad).all().item()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    assert gradients_finite, "Non-finite gradients"

    print("\nBACKPROPAGATION")
    print("Gradients calculated:", gradients_exist)
    print("Gradients finite:", gradients_finite)

    print("\nALL CTC TESTS PASSED")


if __name__ == "__main__":
    test_ctc()
