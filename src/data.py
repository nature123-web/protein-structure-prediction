"""Datasets for distogram prediction.

Two sources are supported:

* ``SyntheticProteinDataset`` -- generates sequences together with Cbeta
  coordinates sampled from a chain model that produces helix-, sheet- and
  coil-like segments. The resulting contact maps have the diagonal bands and
  anti-diagonal strand pairings of real proteins, which makes it a genuine
  smoke test of the whole pipeline rather than noise fitting.
* ``PDBDataset`` -- parses real structures with Biotite.

Both yield the same record, so they are interchangeable in ``train.py``.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

# Standard 20 amino acids plus X for unknown. Index 0 is reserved for padding.
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWYX"
AA_TO_IDX = {aa: i + 1 for i, aa in enumerate(AMINO_ACIDS)}
PAD_IDX = 0
VOCAB_SIZE = len(AMINO_ACIDS) + 1

THREE_TO_ONE = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}


def encode_sequence(seq: str) -> torch.Tensor:
    """Map a one-letter amino acid string to integer indices."""
    return torch.tensor(
        [AA_TO_IDX.get(aa.upper(), AA_TO_IDX["X"]) for aa in seq],
        dtype=torch.long,
    )


def decode_sequence(idx: Sequence[int]) -> str:
    inv = {v: k for k, v in AA_TO_IDX.items()}
    return "".join(inv.get(int(i), "X") for i in idx if int(i) != PAD_IDX)


def pairwise_distances(coords: np.ndarray) -> np.ndarray:
    """Euclidean distance matrix for (L, 3) coordinates."""
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))


def distance_to_bins(
    dist: torch.Tensor, num_bins: int, min_dist: float, max_dist: float
) -> torch.Tensor:
    """Discretise distances into ``num_bins`` classes.

    Bin 0 covers everything below ``min_dist``; the last bin is the catch-all
    "further than ``max_dist``" class, which is where the vast majority of pairs
    land. Bins in between are uniform in angstroms.
    """
    edges = torch.linspace(min_dist, max_dist, num_bins - 1, device=dist.device)
    return torch.bucketize(dist, edges)


def bin_centers(num_bins: int, min_dist: float, max_dist: float) -> torch.Tensor:
    """Representative distance for each bin, used to turn logits into angstroms."""
    edges = torch.linspace(min_dist, max_dist, num_bins - 1)
    step = (max_dist - min_dist) / (num_bins - 2)
    centers = torch.cat([
        torch.tensor([min_dist - step / 2]),
        (edges[:-1] + edges[1:]) / 2,
        torch.tensor([max_dist + step / 2]),
    ])
    return centers


@dataclass
class ProteinRecord:
    """A single training example."""

    name: str
    sequence: torch.Tensor       # (L,) long
    distances: torch.Tensor      # (L, L) float, angstroms
    mask: torch.Tensor           # (L,) bool, True where the residue is resolved


# --------------------------------------------------------------------------- #
# Synthetic generator
# --------------------------------------------------------------------------- #

class SyntheticProteinDataset(Dataset):
    """Procedurally generated proteins with realistic contact topology.

    A chain is assembled from randomly chosen secondary-structure segments.
    Helices advance along an axis while rotating (giving the characteristic
    i,i+3/i+4 contacts); strands run nearly straight and are placed adjacent to
    a previously emitted strand so that anti-parallel sheet contacts appear as
    anti-diagonals in the map; coils random-walk between them.
    """

    HELIX_AAS = "AELMQKRH"
    SHEET_AAS = "VIYFWTC"
    COIL_AAS = "GPSNDT"

    def __init__(
        self,
        n_samples: int,
        min_length: int = 60,
        max_length: int = 256,
        seed: int = 0,
    ) -> None:
        self.n_samples = n_samples
        self.min_length = min_length
        self.max_length = max_length
        self.seed = seed

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> ProteinRecord:
        rng = np.random.default_rng(self.seed * 1_000_003 + idx)
        length = int(rng.integers(self.min_length, self.max_length + 1))
        seq_chars: List[str] = []
        coords: List[np.ndarray] = []

        pos = np.zeros(3)
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        strand_anchors: List[np.ndarray] = []

        while len(coords) < length:
            kind = rng.choice(["helix", "sheet", "coil"], p=[0.4, 0.35, 0.25])
            seg_len = int(rng.integers(5, 16))
            seg_len = min(seg_len, length - len(coords))

            if kind == "helix":
                # 3.6 residues/turn, 1.5 A rise, 2.3 A radius -> real helix geometry.
                axis = direction
                ref = np.array([1.0, 0.0, 0.0])
                if abs(np.dot(ref, axis)) > 0.9:
                    ref = np.array([0.0, 1.0, 0.0])
                u = np.cross(axis, ref)
                u /= np.linalg.norm(u)
                v = np.cross(axis, u)
                for i in range(seg_len):
                    angle = 2 * math.pi * i / 3.6
                    offset = 2.3 * (math.cos(angle) * u + math.sin(angle) * v)
                    coords.append(pos + axis * 1.5 * i + offset)
                    seq_chars.append(rng.choice(list(self.HELIX_AAS)))
                pos = pos + axis * 1.5 * seg_len

            elif kind == "sheet":
                if strand_anchors and rng.random() < 0.7:
                    # Pair anti-parallel with an earlier strand: produces the
                    # anti-diagonal stripes that dominate real contact maps.
                    anchor = strand_anchors[int(rng.integers(len(strand_anchors)))]
                    perp = rng.normal(size=3)
                    perp -= perp.dot(direction) * direction
                    perp /= np.linalg.norm(perp)
                    pos = anchor + perp * 4.8
                    direction = -direction
                strand_anchors.append(pos.copy())
                for i in range(seg_len):
                    jitter = rng.normal(scale=0.25, size=3)
                    coords.append(pos + direction * 3.3 * i + jitter)
                    seq_chars.append(rng.choice(list(self.SHEET_AAS)))
                pos = pos + direction * 3.3 * seg_len

            else:  # coil
                for _ in range(seg_len):
                    step = rng.normal(size=3)
                    step /= np.linalg.norm(step)
                    direction = 0.6 * direction + 0.4 * step
                    direction /= np.linalg.norm(direction)
                    pos = pos + direction * 3.8
                    coords.append(pos.copy())
                    seq_chars.append(rng.choice(list(self.COIL_AAS)))

            new_dir = rng.normal(size=3)
            direction = 0.5 * direction + 0.5 * new_dir / np.linalg.norm(new_dir)
            direction /= np.linalg.norm(direction)

        coords_arr = np.stack(coords[:length])
        dist = pairwise_distances(coords_arr)
        return ProteinRecord(
            name=f"synthetic_{idx:05d}",
            sequence=encode_sequence("".join(seq_chars[:length])),
            distances=torch.from_numpy(dist).float(),
            mask=torch.ones(length, dtype=torch.bool),
        )


# --------------------------------------------------------------------------- #
# Real structures
# --------------------------------------------------------------------------- #

class PDBDataset(Dataset):
    """Loads Cbeta coordinates and sequences from a directory of PDB/mmCIF files.

    Glycine has no Cbeta, so its Calpha is substituted -- the standard
    convention in contact prediction benchmarks. Residues missing from the
    density are excluded via the mask rather than dropped, so residue numbering
    stays aligned with the sequence.
    """

    def __init__(
        self,
        data_dir: str | os.PathLike,
        min_length: int = 40,
        max_length: int = 512,
    ) -> None:
        self.paths = sorted(
            p for p in Path(data_dir).iterdir()
            if p.suffix.lower() in {".pdb", ".cif", ".ent"}
        )
        if not self.paths:
            raise FileNotFoundError(f"No .pdb/.cif files found in {data_dir}")
        self.min_length = min_length
        self.max_length = max_length
        self._cache: dict[int, ProteinRecord] = {}

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> ProteinRecord:
        if idx in self._cache:
            return self._cache[idx]
        record = self._parse(self.paths[idx])
        self._cache[idx] = record
        return record

    def _parse(self, path: Path) -> ProteinRecord:
        try:
            import biotite.structure as struc
            import biotite.structure.io as strucio
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "Parsing real structures needs Biotite: pip install biotite"
            ) from exc

        atoms = strucio.load_structure(str(path))
        if atoms.stack_depth() if hasattr(atoms, "stack_depth") else False:
            atoms = atoms[0]
        atoms = atoms[struc.filter_amino_acids(atoms)]
        # Keep a single chain; multi-chain contact maps mix intra/inter signals.
        chain_id = atoms.chain_id[0]
        atoms = atoms[atoms.chain_id == chain_id]

        seq_chars: List[str] = []
        coords: List[np.ndarray] = []
        mask: List[bool] = []
        for res_id in np.unique(atoms.res_id):
            res = atoms[atoms.res_id == res_id]
            three = str(res.res_name[0]).upper()
            seq_chars.append(THREE_TO_ONE.get(three, "X"))
            target = "CA" if three == "GLY" else "CB"
            sel = res[res.atom_name == target]
            if len(sel) == 0:
                sel = res[res.atom_name == "CA"]
            if len(sel) == 0:
                coords.append(np.zeros(3))
                mask.append(False)
            else:
                coords.append(np.asarray(sel.coord[0], dtype=float))
                mask.append(True)

        coords_arr = np.stack(coords)[: self.max_length]
        seq = "".join(seq_chars)[: self.max_length]
        mask_arr = np.array(mask[: self.max_length])
        dist = pairwise_distances(coords_arr)
        return ProteinRecord(
            name=path.stem,
            sequence=encode_sequence(seq),
            distances=torch.from_numpy(dist).float(),
            mask=torch.from_numpy(mask_arr),
        )


# --------------------------------------------------------------------------- #
# Cropping and batching
# --------------------------------------------------------------------------- #

def random_crop(record: ProteinRecord, crop_size: int, rng: np.random.Generator
                ) -> ProteinRecord:
    """Take a contiguous crop so memory stays O(crop^2) instead of O(L^2)."""
    length = record.sequence.shape[0]
    if length <= crop_size:
        return record
    start = int(rng.integers(0, length - crop_size + 1))
    end = start + crop_size
    return ProteinRecord(
        name=record.name,
        sequence=record.sequence[start:end],
        distances=record.distances[start:end, start:end],
        mask=record.mask[start:end],
    )


class CroppedDataset(Dataset):
    """Wraps any protein dataset and applies a random crop per access."""

    def __init__(self, base: Dataset, crop_size: int, seed: int = 0,
                 deterministic: bool = False) -> None:
        self.base = base
        self.crop_size = crop_size
        self.seed = seed
        self.deterministic = deterministic

    def __len__(self) -> int:
        return len(self.base)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> ProteinRecord:
        record = self.base[idx]
        if self.deterministic:
            # Centre crop for validation, so the metric is comparable epoch to epoch.
            length = record.sequence.shape[0]
            if length <= self.crop_size:
                return record
            start = (length - self.crop_size) // 2
            end = start + self.crop_size
            return ProteinRecord(
                name=record.name,
                sequence=record.sequence[start:end],
                distances=record.distances[start:end, start:end],
                mask=record.mask[start:end],
            )
        rng = np.random.default_rng(self.seed * 7919 + idx)
        return random_crop(record, self.crop_size, rng)


def collate(batch: List[ProteinRecord]) -> dict[str, torch.Tensor]:
    """Pad a list of variable-length records into dense tensors."""
    max_len = max(r.sequence.shape[0] for r in batch)
    b = len(batch)
    seqs = torch.full((b, max_len), PAD_IDX, dtype=torch.long)
    dists = torch.zeros(b, max_len, max_len, dtype=torch.float)
    masks = torch.zeros(b, max_len, dtype=torch.bool)
    for i, r in enumerate(batch):
        n = r.sequence.shape[0]
        seqs[i, :n] = r.sequence
        dists[i, :n, :n] = r.distances
        masks[i, :n] = r.mask
    # A pair is supervised only if both of its residues are resolved.
    pair_mask = masks[:, :, None] & masks[:, None, :]
    return {
        "sequence": seqs,
        "distances": dists,
        "mask": masks,
        "pair_mask": pair_mask,
        "names": [r.name for r in batch],  # type: ignore[dict-item]
    }


def build_datasets(cfg: dict, seed: int = 0) -> tuple[Dataset, Dataset]:
    """Construct train/val datasets from the ``data`` section of a config."""
    d = cfg["data"]
    crop = d["crop_size"]
    if d.get("data_dir"):
        full = PDBDataset(d["data_dir"], d["min_length"], d["max_length"])
        n_val = max(1, int(0.1 * len(full)))
        indices = np.random.default_rng(seed).permutation(len(full))
        val_idx, train_idx = indices[:n_val], indices[n_val:]
        train_base = torch.utils.data.Subset(full, train_idx.tolist())
        val_base = torch.utils.data.Subset(full, val_idx.tolist())
    else:
        train_base = SyntheticProteinDataset(
            d["n_synthetic_train"], d["min_length"], d["max_length"], seed=seed
        )
        val_base = SyntheticProteinDataset(
            d["n_synthetic_val"], d["min_length"], d["max_length"], seed=seed + 10_000
        )
    return (
        CroppedDataset(train_base, crop, seed=seed),
        CroppedDataset(val_base, crop, seed=seed, deterministic=True),
    )
