"""Post-training static int8 quantization and TorchScript export.

Validates host-side int8 inference (size reduction + float-vs-int8 agreement).
"""

import json
import os
import tempfile

import numpy as np
import torch

try:
    import torch.ao.quantization as aq
except Exception:
    import torch.quantization as aq

from data import CLASS_NAMES, FEATURES, ID_TO_LABEL, LABEL_TO_ID, standardize, xy
from model import TinyStressNet, TinyStressNetQuantizable


def _kb(path):
    return os.path.getsize(path) / 1024.0


def _resolve_engine(requested):
    """Pick a quantization backend supported on this platform.

    qnnpack exists on ARM (Apple Silicon, mobile); fbgemm/x86 on Intel/AMD.
    'auto' (or an unsupported request) falls back to the best available one.
    """
    supported = [e for e in torch.backends.quantized.supported_engines if e != "none"]
    if requested and requested != "auto" and requested in supported:
        return requested
    for cand in ("x86", "fbgemm", "qnnpack", "onednn"):
        if cand in supported:
            return cand
    raise RuntimeError(f"No supported quantization engine found; available: {supported}")


def quantize_and_report(df, float_model: TinyStressNet, engine="auto",
                        calib_batches=10, outdir="."):
    """Return a dict of sizes + float/int8 agreement; write TorchScript artifacts."""
    engine = _resolve_engine(engine)
    print(f"Quantization engine: {engine}")
    torch.backends.quantized.engine = engine
    float_model.eval()

    X, _ = xy(df)
    _, X_std = standardize(X)
    calib = torch.tensor(X_std, dtype=torch.float32).unsqueeze(-1)

    # float state-dict and TorchScript sizes
    os.makedirs(outdir, exist_ok=True)
    sd_path = os.path.join(outdir, "tinystressnet_state_dict.pt")
    torch.save(float_model.state_dict(), sd_path)

    example = torch.randn(1, 3, 1)
    float_ts_path = os.path.join(outdir, "tinystressnet_float_ts.pt")
    torch.jit.trace(float_model, example).save(float_ts_path)

    # quantizable twin <- float weights
    q = TinyStressNetQuantizable()
    q.load_state_dict(float_model.state_dict(), strict=False)
    q.eval()
    q_fused = aq.fuse_modules(q, [
        ["features.0", "features.1", "features.2"],
        ["features.4", "features.5", "features.6"],
        ["features.8", "features.9", "features.10"],
    ], inplace=False)
    q_fused.qconfig = aq.get_default_qconfig(engine)
    prepared = aq.prepare(q_fused, inplace=False)
    with torch.no_grad():
        for i in range(0, len(calib), 32):
            prepared(calib[i:i + 32])
            if i // 32 + 1 >= calib_batches:
                break
    quantized = aq.convert(prepared, inplace=False)
    quantized.eval()

    int8_ts_path = os.path.join(outdir, "tinystressnet_int8_ts.pt")
    torch.jit.trace(quantized, example).save(int8_ts_path)

    with torch.no_grad():
        float_pred = float_model(calib).argmax(1).numpy()
        int8_pred = quantized(calib).argmax(1).numpy()
    agreement = float(np.mean(float_pred == int8_pred))

    meta = {
        "class_names": CLASS_NAMES, "label_to_id": LABEL_TO_ID, "id_to_label": ID_TO_LABEL,
        "feature_order": FEATURES,
    }
    with open(os.path.join(outdir, "preprocess_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    sizes = {
        "float_state_dict_kb": _kb(sd_path),
        "float_torchscript_kb": _kb(float_ts_path),
        "int8_torchscript_kb": _kb(int8_ts_path),
    }
    sizes["int8_reduction_vs_float_ts_pct"] = 100.0 * (
        1 - sizes["int8_torchscript_kb"] / sizes["float_torchscript_kb"]
    )
    return {"sizes_kb": sizes, "float_int8_agreement": agreement}
