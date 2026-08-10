# Protein Structure Prediction with Deep Learning

Predict inter-residue **distograms** and **contact maps** from a raw amino-acid
sequence using a transformer encoder coupled to a residual 2D convolution tower.
The design follows the single-sequence branch of the AlphaFold / trRosetta family
of models, scaled down so it trains on a laptop CPU.

```
sequence (L,)
   │  embedding + positional encoding
   ▼
Transformer encoder            → per-residue features (L, d)
   │  outer product + concat
   ▼
Pairwise tensor (L, L, 2d+1)
   │  residual dilated conv tower
   ▼
Distogram logits (bins, L, L)  → argmax → distance / contact map
```

## Why a distogram?

Regressing 3D coordinates directly is unstable because the target is only defined
up to a rigid-body transform. Predicting the **matrix of pairwise Cβ–Cβ distances**
sidesteps that entirely: it is rotation- and translation-invariant, and a 3D
structure can be recovered from it afterwards with gradient descent or MDS. We
discretise distance into `num_bins` bins (2–22 Å plus a "no contact" bin) and
treat the problem as per-pair classification, which trains far more stably than
regression.

A **contact** is conventionally defined as Cβ–Cβ distance < 8 Å, so the contact
probability is just the sum of the probability mass in the bins below 8 Å.

## Install

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quick start

Train on the built-in synthetic protein generator (no download needed — it
produces plausible secondary-structure-like contact patterns so the pipeline can
be validated end to end):

```bash
python -m src.train --config configs/base.yaml
```

Train on real structures — point it at a directory of PDB files:

```bash
python -m src.train --config configs/base.yaml --data-dir data/pdb
```

Predict a contact map for a single sequence:

```bash
python -m src.predict \
    --checkpoint runs/base/best.pt \
    --sequence MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ \
    --out contact_map.png
```

## Getting real training data

The model consumes `(sequence, Cβ coordinates)` pairs. Two easy sources:

1. **PDB files** — download a curated, redundancy-filtered set from
   [PISCES](https://dunbrack.fccc.edu/pisces/) and drop the `.pdb` files into
   `data/pdb/`. `src/data.py` parses them with Biotite.
2. **CASP targets** — useful as a held-out test set, since they are by
   construction unseen structures.

```bash
mkdir -p data/pdb && cd data/pdb
wget https://files.rcsb.org/download/1UBQ.pdb
```

## Metrics

Contact prediction is scored on the **top-L/k long-range contacts**, because
a protein of length L has ~L true contacts but L² candidate pairs — plain
accuracy would be ~99% for a model that predicts "no contact" everywhere.

| Metric | Meaning |
| --- | --- |
| `P@L` | precision of the L highest-confidence predicted contacts |
| `P@L/2`, `P@L/5` | same, stricter — the standard CASP reporting points |
| long-range | only pairs with sequence separation ≥ 24 |
| medium-range | separation 12–23 |

Long-range contacts are the ones that actually determine the fold, so `P@L/5`
long-range is the headline number.

## Configuration

All hyperparameters live in `configs/base.yaml`:

```yaml
model:
  d_model: 128          # per-residue embedding width
  n_heads: 4
  n_encoder_layers: 4   # 1D transformer depth
  n_conv_blocks: 8      # 2D residual tower depth
  num_bins: 24
train:
  epochs: 20
  batch_size: 4
  lr: 0.001
  crop_size: 128        # random crop; keeps memory O(crop²) not O(L²)
```

Cropping matters: the pairwise tensor is quadratic in length, so a 500-residue
protein at `d_model=128` would need several GB. Random 128×128 crops keep memory
flat while still seeing every region of every protein across epochs.

## Layout

```
src/
  data.py       synthetic generator, PDB parsing, cropping, batching
  model.py      ProteinContactNet
  metrics.py    top-L/k precision, per-range breakdown
  train.py      training loop, checkpointing, early stopping
  predict.py    inference + contact map rendering
tests/          pytest suite (shapes, invariances, metric correctness)
```

## Limitations

This is a single-sequence model. Production predictors gain most of their
accuracy from **multiple sequence alignments** — the coevolution signal across
homologous sequences is what pins down long-range contacts. Adding an MSA
encoder (or ESM-2 embeddings as a drop-in replacement for the 1D encoder) is the
single highest-impact extension; `src/model.py` exposes `embed_dim` so a
pretrained language-model embedding can be substituted without touching the
2D tower.

## License

MIT
