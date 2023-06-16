# PCF CUDA Kernel:
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022-2023 Apple Inc. All Rights Reserved.
#

# Unit tests for the CUDA Kernel, saving the code here to ease future improvements on the unit tests

import time
import torch
import pcf_cuda


def index_points(points, idx):
    """

    Input:
        points: input points data, shape [B, N, C]
        idx: sample index data, shape [B, S] / [B, S, K]
    Return:
        new_points:, indexed points data, shape [B, S, C] / [B, S, K, C]
    """
    device = points.device
    BB = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(BB, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


data = torch.load('test_files.pth')
guidance = data['guidance']
guidance = guidance.permute(0, 3, 1, 2).contiguous()
linear_weight = data['linear_weight'][:, :768].contiguous()
weightnet = data['weightnet_out'].permute(0, 3, 2, 1).contiguous()
input_feat = data['layer1_out'].contiguous().to('cuda:0')
neighbor_inds = data['neighbor_inds'].to('cuda:0')
weightnet = weightnet.to('cuda:0')
linear_weight = linear_weight.to('cuda:0')
guidance = guidance.to('cuda:0')

forward = 0
backward = 0
forward_pconv = 0
reps = 1000
discard = 50

print(input_feat.shape)
print(neighbor_inds.shape)
print(guidance.shape)
print(weightnet.shape)
weightnet2 = weightnet.permute(0, 1, 3, 2).contiguous()
gathered_feat = index_points(input_feat, neighbor_inds)
for i in range(reps+discard):
    torch.cuda.synchronize()
    start = time.time()
    output = pcf_cuda.pcf_forward(input_feat, neighbor_inds, guidance, weightnet2)
    # ,linear_weight, torch.zeros(1,dtype=torch.float32).to('cuda:0'))
    torch.cuda.synchronize()
    end = time.time()
    if i > discard:
        forward += end - start
    start = end
    grad_input, grad_guidance, grad_weights = pcf_cuda.pcf_backward(torch.ones_like(output), input_feat, neighbor_inds, guidance, weightnet2)
    torch.cuda.synchronize()
    end = time.time()
    if i > discard:
        backward += end - start
    start = end
    output_pconv = pcf_cuda.pconv_forward(input_feat, neighbor_inds, weightnet2, gathered_feat)
    torch.cuda.synchronize()
    end = time.time()
    if i > discard:
        forward_pconv += end - start
print(output.shape)
print('CUDA Kernel Forward: ', forward / reps * 1000, 'ms')
print('CUDA Kernel Backward: ', backward / reps * 1000, 'ms')
print('CUDA Kernel Forward Pconv: ', forward_pconv / reps * 1000, 'ms')

print('Input grad shape: ', grad_input.shape)
print('Guidance grad shape: ', grad_guidance.shape)
print('Weights grad shape: ', grad_weights.shape)

forward = 0
forward1 = 0
forward2 = 0
backward1 = 0
B = input_feat.shape[0]
K = neighbor_inds.shape[2]
M = neighbor_inds.shape[1]
num_heads = guidance.shape[2]
guidance = guidance.permute(0, 2, 3, 1).contiguous()
# weightnet = weightnet.permute(0,1,3,2).contiguous()
input_feat.requires_grad_(True)
input_feat.retain_grad()
guidance.requires_grad_(True)
weightnet.requires_grad_(True)
guidance.retain_grad()
weightnet.retain_grad()
for i in range(reps+discard):
    torch.cuda.synchronize()
    start = time.time()
    gathered_feat = index_points(input_feat, neighbor_inds)
    new_feat = gathered_feat.permute(0, 3, 2, 1)
    new_feat = new_feat.view(B, -1, num_heads, K, M)
    new_feat_inm = (new_feat * guidance).view(B, -1, K, M).contiguous()
    new_feat2 = torch.matmul(input=new_feat_inm.permute(0, 3, 1, 2).contiguous(), other=weightnet)
    new_feat2 = new_feat2.view(new_feat2.shape[0], new_feat2.shape[1], -1)
    torch.cuda.synchronize()
    t1 = time.time()
    if i > discard:
        forward1 += t1 - start
    fin_func = torch.sum(new_feat2)
    fin_func.backward()
    torch.cuda.synchronize()
    t2 = time.time()
    gathered_feat2 = torch.cat([gathered_feat, gathered_feat], dim=-1)
    new_pconv = torch.matmul(input=gathered_feat2.permute(0, 1, 3, 2).contiguous(), other=weightnet)
    new_pconv = new_pconv.view(new_pconv.shape[0], new_pconv.shape[1], -1)
    torch.cuda.synchronize()
    t3 = time.time()
    grad_input2 = input_feat.grad.clone().detach()
    grad_guidance2 = guidance.grad.clone().detach()
    grad_weights2 = weightnet.grad.clone().detach()
    guidance.grad.data.zero_()
    weightnet.grad.data.zero_()
    input_feat.grad.data.zero_()
    if i > discard:
        backward1 += t2 - t1
        forward2 += t3 - t2
grad_guidance2 = grad_guidance2.permute(0, 3, 1, 2)
# Shape between C_mid and K is different between the 2 versions
grad_weights = grad_weights.permute(0, 1, 3, 2)
print('input grad shape: ', grad_input2.shape)
print('guidance grad shape: ', grad_guidance2.shape)
print('weight grad shape: ', grad_weights2.shape)

print('Pytorch Forward1: ', forward1 / reps * 1000, 'ms')
print('Pytorch backward1: ', backward1 / reps * 1000, 'ms')
print('Pytorch Forward Pconv matmul only: ', forward2 / reps * 1000, 'ms')
print('diff of forward: ', torch.linalg.norm(new_feat2 - output))
print('diff of pconv: ', torch.linalg.norm(new_pconv - output_pconv))
print('diff of input grad: ', torch.linalg.norm(grad_input2 - grad_input))
print('diff of guidance grad: ', torch.linalg.norm(grad_guidance2 - grad_guidance))
print('diff of weight grad: ', torch.linalg.norm(grad_weights2 - grad_weights))
