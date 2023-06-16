# PCF CUDA Kernel:
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022-2023 Apple Inc. All Rights Reserved.
#

# Example code using PCF CUDA Kernel

import torch

import pcf_cuda


class PCFFunction(torch.autograd.Function):
    # Currently, PCF CUDA kernel is by default not used since it leads to some mysterious training degradations, however, it works a bit faster than the non-CUDA
    # implementation at testing time and should be used to achieve the speed benchmark performance
    @staticmethod
    def forward(ctx, input_feat, neighbor_inds, guidance, weightnet):
        # Make sure we are not computing gradient on neighbor_inds
        neighbor_inds.requires_grad = False
        output = pcf_cuda.forward(input_feat, neighbor_inds, guidance, weightnet)
        ctx.save_for_backward({input_feat, neighbor_inds, guidance, weightnet})
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input, grad_guidance, grad_weight = pcf_cuda.backward(grad_output.contiguous(), *ctx.saved_tensors)
        return grad_input, None, grad_guidance, grad_weight


class PCF(torch.nn.Module):
    def __init__(self):
        super(PCF, self).__init__()

    def forward(self, input_features, neighbor_inds, guidance, weightnet):
        return PCFFunction.apply(input_features, neighbor_inds, guidance, weightnet)
