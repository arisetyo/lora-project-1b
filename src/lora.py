import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    Wraps an nn.Linear layer with a LoRA branch.

    Forward pass:
        h = linear(x) + (lora_B @ lora_A @ x^T)^T * (alpha / r)

    The wrapped linear's parameters are frozen. Only lora_A and lora_B
    are trained.
    """

    def __init__(self, linear: nn.Linear, r: int, alpha: float):
        """Initialize LoRA rank/scale parameters and freeze base linear weights."""
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scale = alpha / r

        # Freeze the original layer
        self.linear = linear
        for param in self.linear.parameters():
            param.requires_grad = False

        in_features = linear.in_features
        out_features = linear.out_features

        # LoRA matrices: A projects down to rank, B projects back up
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        # A initialized with Kaiming uniform; B starts at zero so the
        # branch contributes nothing at the start of training
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return base linear output plus scaled low-rank LoRA adaptation."""
        base_out = self.linear(x)
        # x shape: (..., in_features)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T
        return base_out + lora_out * self.scale


def apply_lora_to_gpt2(model, r: int, alpha: float):
    """
    Replaces the c_attn (combined Q/K/V projection) layer in every
    GPT-2 transformer block with a LoRALinear wrapper.

    Returns the modified model.
    """
    for block in model.transformer.h:
        original = block.attn.c_attn
        # GPT-2 uses Conv1D internally; wrap it via a helper that exposes
        # it as a standard linear interface
        block.attn.c_attn = _wrap_conv1d_as_lora(original, r, alpha)
    return model


def _wrap_conv1d_as_lora(conv1d_layer, r: int, alpha: float):
    """
    GPT-2's attention uses transformers.pytorch_utils.Conv1D, which is
    transposed relative to nn.Linear (weight shape is [in, out] not [out, in]).
    This wrapper handles that convention difference.
    """
    # Create an equivalent nn.Linear so LoRALinear can wrap it normally
    in_features = conv1d_layer.weight.shape[0]
    out_features = conv1d_layer.weight.shape[1]

    linear = nn.Linear(in_features, out_features, bias=conv1d_layer.bias is not None)
    with torch.no_grad():
        linear.weight.copy_(conv1d_layer.weight.T)
        if conv1d_layer.bias is not None:
            linear.bias.copy_(conv1d_layer.bias)

    return LoRALinear(linear, r=r, alpha=alpha)


def print_trainable_params(model) -> None:
    """Prints the number and percentage of trainable parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100 * trainable / total
    print(f"Trainable parameters: {trainable:,} / {total:,} ({pct:.3f}%)")
