"""Training loop for ProteinContactNet.

    python -m src.train --config configs/base.yaml
    python -m src.train --config configs/base.yaml --data-dir data/pdb --epochs 50
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import bin_centers, build_datasets, collate, distance_to_bins
from .metrics import evaluate_contacts
from .model import ProteinContactNet, distogram_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(spec: str) -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def lr_lambda_factory(warmup_steps: int, total_steps: int):
    """Linear warmup then cosine decay.

    Warmup matters for transformers: without it the first few large updates
    tend to collapse the attention maps to uniform and the model never recovers.
    """
    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * min(1.0, progress)))
    return fn


def run_epoch(model, loader, cfg, device, optimizer=None, scheduler=None):
    """One pass over ``loader``. Trains when ``optimizer`` is given, else evals."""
    training = optimizer is not None
    model.train(training)
    m = cfg["model"]
    centers = bin_centers(m["num_bins"], m["min_dist"], m["max_dist"]).to(device)

    totals = {"loss": 0.0, "n": 0}
    metric_sums: dict[str, list[float]] = {}

    for batch in tqdm(loader, desc="train" if training else "val", leave=False):
        seq = batch["sequence"].to(device)
        dist = batch["distances"].to(device)
        pair_mask = batch["pair_mask"].to(device)

        target = distance_to_bins(dist, m["num_bins"], m["min_dist"], m["max_dist"])

        with torch.set_grad_enabled(training):
            logits = model(seq, mask=batch["mask"].to(device))
            loss = distogram_loss(logits, target, pair_mask)

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        totals["loss"] += float(loss.detach()) * seq.shape[0]
        totals["n"] += seq.shape[0]

        if not training:
            with torch.no_grad():
                probs = F.softmax(logits, dim=1)
                contact_prob = (probs * (centers < 8.0).view(1, -1, 1, 1)).sum(1)
                batch_metrics = evaluate_contacts(contact_prob, dist, pair_mask)
            for k, v in batch_metrics.items():
                if v == v:
                    metric_sums.setdefault(k, []).append(v)

    out = {"loss": totals["loss"] / max(1, totals["n"])}
    for k, vals in metric_sums.items():
        out[k] = sum(vals) / len(vals)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--data-dir", default=None,
                        help="Directory of PDB files; overrides the config.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    # utf-8-sig: Windows editors and PowerShell redirection add a BOM, which
    # would otherwise turn the first key into "﻿seed".
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8-sig"))
    if args.data_dir:
        cfg["data"]["data_dir"] = args.data_dir
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.out_dir:
        cfg["out_dir"] = args.out_dir

    set_seed(cfg["seed"])
    device = resolve_device(cfg["train"]["device"])
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  out_dir={out_dir}")

    train_ds, val_ds = build_datasets(cfg, seed=cfg["seed"])
    train_loader = DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
        collate_fn=collate, num_workers=cfg["train"]["num_workers"],
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
        collate_fn=collate, num_workers=cfg["train"]["num_workers"],
    )
    print(f"train={len(train_ds)} proteins  val={len(val_ds)} proteins")

    m = cfg["model"]
    model = ProteinContactNet(
        d_model=m["d_model"], n_heads=m["n_heads"],
        n_encoder_layers=m["n_encoder_layers"],
        dim_feedforward=m["dim_feedforward"],
        n_conv_blocks=m["n_conv_blocks"], conv_channels=m["conv_channels"],
        dilations=tuple(m["dilations"]), num_bins=m["num_bins"],
        dropout=m["dropout"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params/1e6:.2f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    total_steps = cfg["train"]["epochs"] * max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda_factory(cfg["train"]["warmup_steps"], total_steps)
    )

    history, best_val, patience = [], float("inf"), 0
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        train_stats = run_epoch(model, train_loader, cfg, device, optimizer, scheduler)
        val_stats = run_epoch(model, val_loader, cfg, device)

        headline = val_stats.get("P@L_5_long", float("nan"))
        print(
            f"epoch {epoch:3d}  train_loss {train_stats['loss']:.4f}  "
            f"val_loss {val_stats['loss']:.4f}  P@L/5(long) {headline:.3f}"
        )
        history.append({"epoch": epoch, "train": train_stats, "val": val_stats})
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

        if val_stats["loss"] < best_val:
            best_val, patience = val_stats["loss"], 0
            torch.save(
                {"model": model.state_dict(), "config": cfg, "epoch": epoch},
                out_dir / "best.pt",
            )
            print(f"  saved new best (val_loss {best_val:.4f})")
        else:
            patience += 1
            if patience >= cfg["train"]["early_stopping_patience"]:
                print(f"early stopping after {epoch} epochs")
                break

    torch.save({"model": model.state_dict(), "config": cfg}, out_dir / "last.pt")
    print(f"done. best val_loss={best_val:.4f}")


if __name__ == "__main__":
    main()
