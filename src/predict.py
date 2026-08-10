"""Predict and visualise a contact map for a single sequence.

    python -m src.predict --checkpoint runs/base/best.pt \
        --sequence MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ --out contact_map.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import bin_centers, encode_sequence
from .model import ProteinContactNet


def load_model(checkpoint: str | Path, device: torch.device
               ) -> tuple[ProteinContactNet, dict]:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    m = cfg["model"]
    model = ProteinContactNet(
        d_model=m["d_model"], n_heads=m["n_heads"],
        n_encoder_layers=m["n_encoder_layers"],
        dim_feedforward=m["dim_feedforward"],
        n_conv_blocks=m["n_conv_blocks"], conv_channels=m["conv_channels"],
        dilations=tuple(m["dilations"]), num_bins=m["num_bins"],
        dropout=m["dropout"],
    )
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, cfg


@torch.no_grad()
def predict(model: ProteinContactNet, sequence: str, cfg: dict,
            device: torch.device) -> dict[str, np.ndarray]:
    """Return contact probabilities and expected distances for a sequence."""
    m = cfg["model"]
    seq = encode_sequence(sequence).unsqueeze(0).to(device)
    logits = model(seq)
    probs = F.softmax(logits, dim=1)[0]                       # (bins, L, L)
    centers = bin_centers(m["num_bins"], m["min_dist"], m["max_dist"]).to(device)

    contact = (probs * (centers < 8.0).view(-1, 1, 1)).sum(0)
    # Expected distance under the predicted distribution -- a better point
    # estimate than the argmax bin, which quantises hard at bin width.
    expected = (probs * centers.view(-1, 1, 1)).sum(0)
    return {
        "contact_prob": contact.cpu().numpy(),
        "expected_distance": expected.cpu().numpy(),
    }


def plot(result: dict[str, np.ndarray], sequence: str, out_path: str | Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    im0 = axes[0].imshow(result["contact_prob"], cmap="viridis", vmin=0, vmax=1)
    axes[0].set_title(f"Contact probability (< 8 Å)\n{len(sequence)} residues")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(result["expected_distance"], cmap="magma_r")
    axes[1].set_title("Expected Cβ–Cβ distance (Å)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    for ax in axes:
        ax.set_xlabel("residue j")
        ax.set_ylabel("residue i")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def read_fasta(path: str | Path) -> str:
    lines = Path(path).read_text().splitlines()
    return "".join(l.strip() for l in lines if l and not l.startswith(">"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sequence", default=None, help="Raw amino-acid string.")
    parser.add_argument("--fasta", default=None, help="Path to a FASTA file.")
    parser.add_argument("--out", default="contact_map.png")
    parser.add_argument("--save-npz", default=None,
                        help="Also dump the raw matrices to this .npz path.")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if not args.sequence and not args.fasta:
        parser.error("provide --sequence or --fasta")
    sequence = args.sequence or read_fasta(args.fasta)

    device = torch.device(args.device)
    model, cfg = load_model(args.checkpoint, device)
    result = predict(model, sequence, cfg, device)

    n_contacts = int((result["contact_prob"] > 0.5).sum() // 2)
    print(f"sequence length: {len(sequence)}")
    print(f"predicted contacts (p > 0.5): {n_contacts}")

    plot(result, sequence, args.out)
    if args.save_npz:
        np.savez_compressed(args.save_npz, sequence=sequence, **result)
        print(f"wrote {args.save_npz}")


if __name__ == "__main__":
    main()
