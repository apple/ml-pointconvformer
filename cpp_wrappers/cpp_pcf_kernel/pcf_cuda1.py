import torch

import pcf_cuda

class PCFFunction(torch.autograd.Function):
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
