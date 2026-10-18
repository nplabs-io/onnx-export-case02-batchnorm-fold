# PyTorch → ONNX: a ResNet-18 that exports cleanly and computes the wrong numbers

With PyTorch 2.9 and its default (dynamo) ONNX exporter, a ResNet-18 classifier
exports without a single error or warning, loads in ONNX Runtime, runs, and
returns different outputs from the PyTorch model on the same input.

This repo reproduces the problem reported in
[microsoft/onnxruntime#26504](https://github.com/microsoft/onnxruntime/issues/26504),
finds where it happens, and measures three ways to fix it.

**Short answer:** the bug is in the graph optimizer of `onnxscript 0.5.6`, which
the exporter runs after export. It folds every BatchNorm layer into the preceding
convolution and gets the numbers slightly wrong. `onnxscript 0.7.1` performs the
same folding correctly.

## Results

Same model, same seeded weights, same input tensor fed to PyTorch and to ONNX Runtime
(CPU). A run passes when cosine similarity ≥ 0.9999 **and** the relative max absolute
difference ≤ 1e-4.

| torch | onnxscript | export settings | max abs diff | cosine | result |
|---|---|---|---|---|---|
| 2.9.0 | 0.5.6 | defaults (dynamo, `optimize=True`) | 2.7e-2 | 0.9999994 | **mismatch** |
| 2.9.0 | 0.5.6 | `dynamo=False` | 5.0e-6 | 0.9999999 | match |
| 2.9.0 | 0.5.6 | `optimize=False` | 3.8e-6 | 1.0000000 | match |
| 2.9.0 | 0.7.1 | defaults | 5.0e-6 | 1.0000000 | match |
| 2.11.0 | 0.7.1 | defaults | 4.3e-6 | 1.0000000 | match |

Raw console output for every row is in [`results/`](results/).

Note the cosine similarity of the broken model: **0.9999994**. A check that only
looks at cosine similarity passes this model. The error is visible only in the
absolute difference.

## Where it breaks

`narrow.py` exports the same model twice in the broken stack, once with the
optimizer and once without, and counts the operators in each graph:

| ONNX op | `optimize=False` | `optimize=True` |
|---|---|---|
| BatchNormalization | 20 | 0 |
| Shape / Expand / CastLike / Constant | 20 / 40 / 20 / 25 | 0 |

- `optimize=False`: all 20 BatchNorm layers kept, outputs **match**.
- `optimize=True`: all 20 BatchNorm layers folded into the convolutions, outputs **mismatch**.
- Same test with `onnxscript 0.7.1` (PyTorch unchanged): the optimizer makes the
  **same** change (20 → 0 BatchNorm), and the outputs **match**.

So the folding step itself is fine; its arithmetic in `onnxscript 0.5.6` was not.
The optimizer applies 40 rewrite rules in total; the individual rule was not isolated.

## Fixes (each measured above)

1. Upgrade onnxscript: `pip install -U onnxscript`. 0.7.1 was tested; no PyTorch
   upgrade is needed.
2. Keep your versions and skip the optimizer:
   `torch.onnx.export(model, args, path, optimize=False)`.
3. Use the legacy exporter: `torch.onnx.export(model, args, path, dynamo=False)`.

Whatever you choose, compare the ONNX outputs with PyTorch on the same input and
check the absolute difference, not only cosine similarity.

## Reproduce

Tested on Windows 11, Python 3.11, CPU only. The model is built in code; nothing is
downloaded except the Python packages.

```bash
# broken stack (state at the time of the issue)
python -m venv .venv-broken
.venv-broken\Scripts\python -m pip install -r requirements-broken.txt
.venv-broken\Scripts\python repro.py     # expect A and C: MISMATCH, B: MATCH
.venv-broken\Scripts\python narrow.py    # expect optimize=True MISMATCH, optimize=False MATCH

# same PyTorch, newer onnxscript
python -m venv .venv-fixed
.venv-fixed\Scripts\python -m pip install -r requirements-fixed.txt
.venv-fixed\Scripts\python narrow.py     # expect both MATCH
```

On Linux/macOS use `.venv-broken/bin/python` instead of `.venv-broken\Scripts\python`.
Each run writes `.onnx` files (~45 MB each) to the working folder; they are ignored by git.

| File | What it does |
|---|---|
| `repro.py` | The reporter's model and quadrant wrapper. A) full wrapper with export defaults, B) `dynamo=False`, C) model only, preprocessing removed |
| `narrow.py` | Same model; `optimize=True` vs `optimize=False`, plus the operator count diff |
| `requirements-broken.txt` | PyTorch 2.9.0 + onnxscript 0.5.6 (early November 2025) |
| `requirements-fixed.txt` | PyTorch 2.9.0 + onnxscript 0.7.1 |

## Limits

- **Weights.** The reporter's trained checkpoint is not public. The model uses seeded
  random weights with randomized BatchNorm statistics. The difference measured here
  (2.7e-2) is smaller than what the reporter saw with trained weights; the mechanism
  is the same, the size is not.
- **Fixing version.** The onnxscript release that fixed the bug was not bisected;
  only 0.5.6 (broken) and 0.7.1 (fixed) were measured.
- **Scope.** CPU execution provider only. Task-level accuracy was not measured; this
  repo checks numerical equivalence of the raw outputs.

## License

MIT
