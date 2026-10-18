"""
Narrowing step for microsoft/onnxruntime#26504.

Same model as repro.py (tiles-only wrapper, variant C).
Exports twice with the default (dynamo) exporter:
  1) optimize=True  (default)
  2) optimize=False (onnxscript graph optimizer skipped)
Then compares each with PyTorch and prints which op types the optimizer changed.
"""

import sys
from collections import Counter

import numpy as np
import torch  # torch BEFORE onnxruntime

import onnx
import onnxruntime as ort

import repro as R


def run_onnx(path, x):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return sess.run(None, {sess.get_inputs()[0].name: x.numpy()})[0].reshape(-1)


def op_counts(path):
    return Counter(n.op_type for n in onnx.load(path).graph.node)


def check(label, ref, got):
    diff = float(np.abs(ref - got).max())
    scale = float(np.abs(ref).max()) or 1.0
    cos = float(np.dot(ref, got) / (np.linalg.norm(ref) * np.linalg.norm(got) + 1e-12))
    ok = cos >= R.COS_MIN and diff / scale <= R.REL_MAX
    print(f"  {label:14s} onnx logits {np.array2string(got, precision=4)}")
    print(f"  {'':14s} max abs diff {diff:.3e} | cosine {cos:.8f} -> {'MATCH' if ok else 'MISMATCH'}")
    return ok


def main():
    from importlib.metadata import version
    print(f"torch {torch.__version__} | onnxscript {version('onnxscript')} | "
          f"onnx-ir {version('onnx-ir')} | onnxruntime {ort.__version__}")

    model = R.load_model()
    full = R.QuadrantWrapper(model).eval()
    tiles_w = R.TilesWrapper(model).eval()
    g = torch.Generator().manual_seed(R.SEED)
    image = torch.randint(0, 256, (1, R.FULL_H, R.FULL_W, 3), dtype=torch.uint8, generator=g)
    with torch.no_grad():
        tiles = full.preprocess(image).contiguous()
        ref = tiles_w(tiles).numpy().reshape(-1)
    print(f"  pytorch        logits {np.array2string(ref, precision=4)}")

    kw = {"input_names": ["input"], "dynamo": True}
    torch.onnx.export(tiles_w, (tiles,), "n_opt_on.onnx", optimize=True, **kw)
    torch.onnx.export(tiles_w, (tiles,), "n_opt_off.onnx", optimize=False, **kw)

    print()
    on_ok = check("optimize=True", ref, run_onnx("n_opt_on.onnx", tiles))
    off_ok = check("optimize=False", ref, run_onnx("n_opt_off.onnx", tiles))

    c_on, c_off = op_counts("n_opt_on.onnx"), op_counts("n_opt_off.onnx")
    print("\n  op type          optimize=False -> optimize=True")
    for op in sorted(set(c_on) | set(c_off)):
        if c_on[op] != c_off[op]:
            print(f"  {op:16s} {c_off[op]:5d} -> {c_on[op]:5d}")

    print(f"\nSUMMARY: optimize=True {'MATCH' if on_ok else 'MISMATCH'} | "
          f"optimize=False {'MATCH' if off_ok else 'MISMATCH'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
