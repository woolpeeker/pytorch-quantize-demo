import torch
from torch.quantization.observer import MinMaxObserver
import math

class ExpScaleMinMaxObserver(MinMaxObserver):
    @torch.jit.export
    def calculate_qparams(self):
        r"""Calculates the quantization parameters."""
        return self._calculate_qparams(self.min_val, self.max_val)
    
    @torch.jit.export
    def _calculate_qparams(self, min_val, max_val):
        # type: (Tensor, Tensor) -> Tuple[Tensor, Tensor]
        r"""Calculates the per tensor quantization parameters, given the min/max.

        Args:
            min_val: Per tensor minimum value
            max_val: Per tensor maximum value

        Returns:
            scale: Scale as a tensor of shape (1,), which is alwasy exp of 2
            zero_point: always 0
        """

        if max_val.numel() == 0 or min_val.numel() == 0:
            warnings.warn("Must run observer before calling calculate_qparams.\
                           Returning default scale and zero point.")
            return torch.tensor([1.0]), torch.tensor([0])

        assert min_val <= max_val, "min {} should be less than max {}".format(
            min_val, max_val
        )

        if self.dtype == torch.qint8:
            if self.reduce_range:
                qmin, qmax = -64, 63
            else:
                qmin, qmax = -128, 127
        else:
            if self.reduce_range:
                qmin, qmax = 0, 127
            else:
                qmin, qmax = 0, 255

        max_val, min_val = float(max_val), float(min_val)
        min_val = min(0.0, min_val)
        max_val = max(0.0, max_val)
        if max_val == min_val:
            scale = 1.0
            zero_point = 0
        else:
            if self.qscheme == torch.per_tensor_symmetric or self.qscheme == torch.per_channel_symmetric:
                max_val = max(-min_val, max_val)
                scale = max_val / ((qmax - qmin) / 2)
                scale = max(scale, self.eps)
                scale = 0.5 ** round(math.log(scale, 0.5))
                zero_point = 0 if self.dtype == torch.qint8 else 128
            else:
                scale = (max_val - min_val) / float(qmax - qmin)
                scale = max(scale, self.eps)
                scale = 0.5 ** round(math.log(scale, 0.5))
                zero_point = qmin - round(min_val / scale)
                zero_point = max(qmin, zero_point)
                zero_point = min(qmax, zero_point)
                zero_point = int(zero_point)

        return torch.tensor([scale]), torch.tensor([zero_point])
