"""TinyStressNet model definitions.

Input is a (batch, 3, 1) tensor of standardized physiological features
(GSR, heart rate, skin temperature). Every convolution uses kernel size 1, so
the network is a small channel-wise MLP with batch normalization, kept in
convolutional form to match embedded inference runtimes.

3,139 trainable parameters; 2,800 convolution + linear MACs at input length 1.
"""

import torch
import torch.nn as nn

try:
    import torch.ao.quantization as aq
except Exception:  # older torch
    import torch.quantization as aq


def _backbone():
    return nn.Sequential(
        nn.Conv1d(3, 16, kernel_size=1), nn.BatchNorm1d(16), nn.ReLU(inplace=True), nn.Dropout(0.2),
        nn.Conv1d(16, 32, kernel_size=1), nn.BatchNorm1d(32), nn.ReLU(inplace=True), nn.Dropout(0.3),
        nn.Conv1d(32, 64, kernel_size=1), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(0.4),
    )


def _head(num_classes):
    return nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(64, num_classes))


class TinyStressNet(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.features = _backbone()
        self.classifier = _head(num_classes)

    def forward(self, x):
        return self.classifier(self.features(x))


class TinyStressNetQuantizable(nn.Module):
    """Quantization-friendly twin used only for post-training int8 export.

    The feature/classifier blocks mirror TinyStressNet exactly, so a trained
    float state_dict loads directly (the quant/dequant stubs are ignored).
    """

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.quant = aq.QuantStub()
        self.features = _backbone()
        self.classifier = _head(num_classes)
        self.dequant = aq.DeQuantStub()

    def forward(self, x):
        return self.dequant(self.classifier(self.features(self.quant(x))))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_macs(length: int = 1) -> int:
    """Convolution + linear multiply-accumulate operations for input length L."""
    return (3 * 16 + 16 * 32 + 32 * 64) * length + 64 * 3  # = 2800 at L=1


def count_params_analytic() -> int:
    """Trainable parameters including batch-norm affine terms (no torch needed)."""
    block = lambda cin, cout: (cin * cout + cout) + 2 * cout  # conv (w+b) + BN (gamma, beta)
    return block(3, 16) + block(16, 32) + block(32, 64) + (64 * 3 + 3)  # = 3139


if __name__ == "__main__":
    print(f"Parameters: {count_parameters(TinyStressNet()):,}")
    print(f"MACs (L=1): {count_macs():,}")
