from typing import Any, Callable

import torch

from research.grad_norm.modules.grad_norm.common import GradLoggingLayer


class CaptureGradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, log_fn: Callable[[str, Any], None]):
        ctx._log_fn = log_fn
        return x

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        ctx._log_fn("raw_grad", grad_out)
        return grad_out, None


class GradLogLayer(GradLoggingLayer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        self.update_cache_for_logging("activations", x)
        return CaptureGradFunction.apply(x, self.update_cache_for_logging)
