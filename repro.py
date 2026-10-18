"""
Repro attempt for microsoft/onnxruntime#26504
"Outputs are completely different between .onnx model and .pth model"

Model and wrapper are copied from the reporter's attached export_to_onnx.py.
His trained weights are not public, so the model uses seeded random weights
(BatchNorm running stats are randomized too, so BN folding bugs would show).

Three checks, each: export -> run in ONNX Runtime (CPU) -> compare with PyTorch
on the SAME input tensor:
  A) full quadrant wrapper, torch.onnx.export defaults (what the reporter did)
  B) full quadrant wrapper, dynamo=False (the reporter's workaround)
  C) tiles-only wrapper, defaults (the reporter's last test: same tiles in, still different)
"""

import platform
import sys

import numpy as np
import torch  # torch BEFORE onnxruntime
from torch import nn
from torchvision import models

import onnxruntime as ort

SEED = 0
FULL_H, FULL_W = 2048, 2448  # reporter's camera frame size
COS_MIN = 0.9999
REL_MAX = 1e-4  # max|diff| / max|ref|


def load_model():
    # Same architecture as the reporter's load_model(), but no download.
    torch.manual_seed(SEED)
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 1))
    g = torch.Generator().manual_seed(SEED)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                n = m.num_features
                m.running_mean.copy_(torch.randn(n, generator=g) * 0.1)
                m.running_var.copy_(torch.rand(n, generator=g) * 0.5 + 0.75)
                m.weight.copy_(torch.rand(n, generator=g) * 0.5 + 0.75)
                m.bias.copy_(torch.randn(n, generator=g) * 0.1)
    return model.eval()


class QuadrantWrapper(nn.Module):
    """Copied from the reporter's export_to_onnx.py (unused target_h/w kept out)."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("mean_imagenet", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std_imagenet", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def preprocess(self, x):
        tiles = []
        tile_h, tile_w = 1024, 1224
        for i in range(2):
            for j in range(2):
                tiles.append(x[:, i * tile_h:(i + 1) * tile_h, j * tile_w:(j + 1) * tile_w, :])
        tiles = torch.cat(tiles, dim=0)
        tiles = tiles.permute(0, 3, 1, 2).float()
        tiles = tiles / 255.0
        return (tiles - self.mean_imagenet) / self.std_imagenet

    def forward(self, image):
        logits = self.model(self.preprocess(image))
        conf = torch.sigmoid(logits)
        preds = (conf > 0.5).to(logits.dtype)
        return logits, preds


class TilesWrapper(nn.Module):
    """The reporter's last test: model only, tiles already preprocessed."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tiles):
        return self.model(tiles)


def export(module, x, path, dynamo):
    kw = {"input_names": ["input"]}
    if dynamo is None:
        torch.onnx.export(module, (x,), path, **kw)  # library default
    else:
        torch.onnx.export(module, (x,), path, dynamo=dynamo, **kw)


def compare(name, module, x, path, dynamo):
    print(f"\n=== {name} ===")
    try:
        export(module, x, path, dynamo)
    except Exception as exc:  # noqa: BLE001
        print(f"  EXPORT FAILED: {type(exc).__name__}: {str(exc)[:300]}")
        return None
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    got = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})[0].reshape(-1)
    with torch.no_grad():
        out = module(x)
    ref = (out[0] if isinstance(out, tuple) else out).numpy().reshape(-1)

    diff = float(np.abs(ref - got).max())
    scale = float(np.abs(ref).max()) or 1.0
    cos = float(np.dot(ref, got) / (np.linalg.norm(ref) * np.linalg.norm(got) + 1e-12))
    ok = cos >= COS_MIN and diff / scale <= REL_MAX
    print(f"  pytorch logits : {np.array2string(ref, precision=4)}")
    print(f"  onnx    logits : {np.array2string(got, precision=4)}")
    print(f"  max abs diff {diff:.3e} | relative {diff / scale:.3e} | cosine {cos:.8f}")
    print(f"  -> {'MATCH' if ok else 'MISMATCH'}")
    return ok


def main():
    print(f"python {platform.python_version()} | {platform.system()} {platform.release()}")
    for pkg in ("torch", "torchvision", "onnx", "onnxscript", "onnxruntime", "onnxruntime-gpu"):
        try:
            from importlib.metadata import version
            print(f"  {pkg:16s} {version(pkg)}")
        except Exception:  # noqa: BLE001
            print(f"  {pkg:16s} -")

    model = load_model()
    full = QuadrantWrapper(model).eval()
    tiles_w = TilesWrapper(model).eval()

    g = torch.Generator().manual_seed(SEED)
    image = torch.randint(0, 256, (1, FULL_H, FULL_W, 3), dtype=torch.uint8, generator=g)
    with torch.no_grad():
        tiles = full.preprocess(image).contiguous()

    results = {
        "A_full_default": compare("A) full wrapper, export defaults", full, image, "a_full_default.onnx", None),
        "B_full_legacy": compare("B) full wrapper, dynamo=False", full, image, "b_full_legacy.onnx", False),
        "C_tiles_default": compare("C) tiles only, export defaults", tiles_w, tiles, "c_tiles_default.onnx", None),
    }
    print("\nSUMMARY:", {k: ("MATCH" if v else "MISMATCH" if v is False else "EXPORT FAILED") for k, v in results.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
