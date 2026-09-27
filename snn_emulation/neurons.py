"""Spiking neuron models used in the software emulation experiments."""

import torch
import torch.nn as nn
from snntorch import surrogate


class AdEx(nn.Module):
    """Adaptive Exponential Integrate-and-Fire neuron with surrogate gradients."""

    def __init__(
        self,
        C=0.25,
        gL=0.1,
        EL=0.0,
        VT=0.5,
        DT=0.5,
        a=0.15,
        b=0.15,
        tau_w=80.0,
        v_reset=0.3,
        v_thresh=2.0,
        dt=1.0,
        surrogate_type="fast_sigmoid",
        surrogate_param=40.0,
    ):
        super().__init__()
        self.C = C
        self.gL = gL
        self.EL = EL
        self.VT = VT
        self.DT = DT
        self.a = a
        self.b = b
        self.tau_w = tau_w
        self.v_reset = v_reset
        self.v_thresh = v_thresh
        self.dt = dt
        self.spike_grad = _make_surrogate(surrogate_type, surrogate_param)

    def forward(self, I_in, V, w):
        dw = (self.a * (V - self.EL) - w) / self.tau_w
        exp_arg = torch.clamp((V - self.VT) / self.DT, max=10.0)
        dV = (
            -self.gL * (V - self.EL)
            + self.gL * self.DT * torch.exp(exp_arg)
            - w
            + I_in
        ) / self.C

        w_next = w + dw * self.dt
        V_next = V + dV * self.dt

        spike_hard = (V_next >= self.v_thresh).float()
        surrogate_value = self.spike_grad(V_next - self.v_thresh)
        spike = spike_hard + surrogate_value - surrogate_value.detach()

        V_next = V_next * (1 - spike) + self.v_reset * spike
        w_next = w_next + self.b * spike
        return spike, V_next, w_next


class LIF(nn.Module):
    """Leaky Integrate-and-Fire baseline with the same surrogate-gradient interface."""

    def __init__(
        self,
        C=0.25,
        gL=0.1,
        EL=0.0,
        v_reset=0.3,
        v_thresh=2.0,
        dt=1.0,
        surrogate_type="fast_sigmoid",
        surrogate_param=40.0,
    ):
        super().__init__()
        self.C = C
        self.gL = gL
        self.EL = EL
        self.v_reset = v_reset
        self.v_thresh = v_thresh
        self.dt = dt
        self.spike_grad = _make_surrogate(surrogate_type, surrogate_param)

    def forward(self, I_in, V):
        dV = (-self.gL * (V - self.EL) + I_in) / self.C
        V_next = V + dV * self.dt

        spike_hard = (V_next >= self.v_thresh).float()
        surrogate_value = self.spike_grad(V_next - self.v_thresh)
        spike = spike_hard + surrogate_value - surrogate_value.detach()

        V_next = V_next * (1 - spike) + self.v_reset * spike
        return spike, V_next


def _make_surrogate(surrogate_type, surrogate_param):
    if surrogate_type == "fast_sigmoid":
        return surrogate.fast_sigmoid(slope=surrogate_param)
    if surrogate_type == "atan":
        return surrogate.atan(alpha=surrogate_param)
    raise ValueError("surrogate_type must be 'fast_sigmoid' or 'atan'")
