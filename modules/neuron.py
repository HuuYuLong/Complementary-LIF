from typing import Callable

import torch
from spikingjelly.clock_driven.neuron import LIFNode as LIFNode_sj

from modules.surrogate import Rectangle


class ComplementaryLIFNeuron(LIFNode_sj):
    def __init__(self, tau: float = 2., decay_input: bool = False, v_threshold: float = 1.,
                 v_reset: float = None, surrogate_function: Callable = Rectangle(),
                 detach_reset: bool = False, cupy_fp32_inference=False, **kwargs):
        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset, cupy_fp32_inference)
        self.register_memory('c', 0.)  # Complementary memory

    def forward(self, x: torch.Tensor):
        self.neuronal_charge(x)                             # LIF charging
        self.c = self.c * torch.sigmoid(self.v / self.tau)  # Forming
        spike = self.neuronal_fire()                        # LIF fire
        self.c += spike                                     # Strengthen
        self.neuronal_reset(spike)                          # LIF reset
        self.v = self.v - spike * torch.sigmoid(self.c)     # Reset
        return spike

    def neuronal_charge(self, x: torch.Tensor):
        self._charging_v(x)

    def neuronal_reset(self, spike: torch.Tensor):
        self._reset(spike)

    def _charging_v(self, x: torch.Tensor):
        if self.decay_input:
            x = x / self.tau

        if self.v_reset is None or self.v_reset == 0:
            if type(self.v) is float:
                self.v = x
            else:
                self.v = self.v * (1 - 1. / self.tau) + x
        else:
            if type(self.v) is float:
                self.v = self.v_reset * (1 - 1. / self.tau) + self.v_reset / self.tau + x
            else:
                self.v = self.v * (1 - 1. / self.tau) + self.v_reset / self.tau + x

    def _reset(self, spike):
        if self.v_reset is None:
            # soft reset
            self.v = self.v - spike * self.v_threshold
        else:
            # hard reset
            self.v = (1. - spike) * self.v + spike * self.v_reset


class MultiStepCLIFNeuron(ComplementaryLIFNeuron):
    def __init__(self, tau: float = 2., decay_input: bool = False, v_threshold: float = 1.,
                 v_reset: float = None, surrogate_function: Callable = Rectangle(),
                 detach_reset: bool = False, cupy_fp32_inference=False, **kwargs):
        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset, cupy_fp32_inference)

    def forward(self, x_seq: torch.Tensor):
        assert x_seq.dim() > 1
        # x_seq.shape = [T, *]
        spike_seq = []
        self.v_seq = []
        for t in range(x_seq.shape[0]):
            spike_seq.append(super().forward(x_seq[t]).unsqueeze(0))
            self.v_seq.append(self.v.unsqueeze(0))
        spike_seq = torch.cat(spike_seq, 0)
        self.v_seq = torch.cat(self.v_seq, 0)
        return spike_seq



if __name__ == '__main__':
    T= 8
    x_input = torch.rand((T, 3, 32, 32)) * 1.2
    clif = ComplementaryLIFNeuron()
    clif_m = MultiStepCLIFNeuron()

    s_list = []
    for t in range(T):
        s = clif(x_input[t])
        s_list.append(s)

    s_list = torch.stack(s_list, dim=0)
    s_output = clif_m(x_input)

    print(s_list.mean())
    print(s_output.mean())
    assert torch.sum(s_output - torch.Tensor(s_list)) == 0