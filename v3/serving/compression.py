"""Model compression: post-training INT8 dynamic quantization and
magnitude-based unstructured pruning, applied to a trained checkpoint's
nn.Linear layers (every attention projection, feed-forward layer, and the
final vocabulary projection). Both are POST-TRAINING -- no retraining/
fine-tuning step -- the fast, "free" end of the compression spectrum, at
the cost of some quality; quantization-aware training or iterative
pruning-with-fine-tuning would recover more quality at a real extra
training cost outside this project's scope.
"""
import copy

import torch
import torch.nn.utils.prune as prune


def quantize_dynamic_int8(model):
    """PyTorch's standard post-training dynamic INT8 quantization
    (torch.ao.quantization.quantize_dynamic) over every nn.Linear layer's
    weights (activations stay float, quantized per-batch at inference --
    "dynamic"). CPU-only: PyTorch's stock quantized INT8 GEMM kernels this
    API dispatches to (fbgemm/qnnpack) have no CUDA implementation, so
    this only ever runs on CPU, NOT this project's GPU serving path --
    documented and benchmarked as such below, not silently glossed over.
    Still a real, relevant technique: smaller weights mean less to load
    into memory / transfer over network regardless of where the model
    ultimately runs, and it's the standard, no-extra-dependency way to
    quantize a PyTorch model post-training."""
    model_cpu = copy.deepcopy(model).cpu().eval()
    return torch.ao.quantization.quantize_dynamic(
        model_cpu, {torch.nn.Linear}, dtype=torch.qint8
    )


def prune_unstructured(model, amount):
    """Magnitude-based unstructured pruning (L1: zero out the
    smallest-magnitude weights) over every nn.Linear layer's weight
    tensor, at the given sparsity fraction (e.g. 0.3 = 30% of weights
    zeroed). Returns a pruned COPY; the original model is untouched.
    prune.remove() bakes the zeros permanently into the weight tensor
    (rather than keeping a separate mask), matching how the pruned model
    would actually be shipped.

    Deliberately NOT claimed to speed up GPU inference here without
    measuring it first: a dense tensor with zeros in it still occupies
    the same memory and still gets multiplied elementwise by a standard
    GEMM kernel -- unstructured sparsity only pays off with a SPARSE
    tensor format and a sparse-aware kernel, neither of which this
    project uses. Whether pruning helps or does nothing at this project's
    scale is measured directly (see the comparison study), not assumed
    either way."""
    model_copy = copy.deepcopy(model)
    for module in model_copy.modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")
    return model_copy


def model_size_mb(model):
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return total_bytes / (1024 ** 2)


def linear_sparsity_pct(model):
    total, zeros = 0, 0
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            total += module.weight.numel()
            zeros += (module.weight == 0).sum().item()
    return 100.0 * zeros / total if total else 0.0
