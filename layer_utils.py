#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022-2023 Apple Inc. All Rights Reserved.
#

import torch
import torch.nn as nn
import torch.nn.functional as F
import pcf_cuda
from util.cp_batchnorm import CpBatchNorm2d


def index_points(points, idx):
    """
    Input:
        points: input points data, shape [B, N, C]
        idx: sample index data, shape [B, S] / [B, S, K]
    Return:
        new_points:, indexed points data, shape [B, S, C] / [B, S, K, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(
        device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


class PCFFunction(torch.autograd.Function):
    '''
    Function for the PCF CUDA kernel
    '''
    @staticmethod
    def forward(ctx, input_feat, neighbor_inds, guidance, weightnet):
        # Make sure we are not computing gradient on neighbor_inds
        neighbor_inds.requires_grad = False
        output = pcf_cuda.pcf_forward(
            input_feat, neighbor_inds, guidance, weightnet)
        ctx.save_for_backward(input_feat, neighbor_inds, guidance, weightnet)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input, grad_guidance, grad_weight = pcf_cuda.pcf_backward(
            grad_output.contiguous(), *ctx.saved_tensors)
        return grad_input, None, grad_guidance, grad_weight


class PCF(torch.nn.Module):
    '''
    This class uses the CUDA kernel to fuse gather -> matrix multiplication in PCF which improves speed.
    Right now, it is numerically correct, but somehow it will mysteriously reduce training accuracy, hence only recommended to use during testing time
    '''

    def __init__(self):
        super(PCF, self).__init__()

    @staticmethod
    def forward(input_features, neighbor_inds, guidance, weightnet):
        return PCFFunction.apply(
            input_features,
            neighbor_inds,
            guidance,
            weightnet)


class PConvFunction(torch.autograd.Function):
    '''
    Function for the PointConv CUDA kernel
    '''
    @staticmethod
    def forward(
            ctx,
            input_feat,
            neighbor_inds,
            weightnet,
            additional_features):
        # Make sure we are not computing gradient on neighbor_inds
        neighbor_inds.requires_grad = False
        output = pcf_cuda.pconv_forward(
            input_feat, neighbor_inds, weightnet, additional_features)
        ctx.save_for_backward(
            input_feat,
            neighbor_inds,
            weightnet,
            additional_features)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input, grad_weight, grad_additional = pcf_cuda.pconv_backward(
            grad_output.contiguous(), *ctx.saved_tensors)
        return grad_input, None, grad_weight, grad_additional


class PConv(torch.nn.Module):
    '''
    This class uses the CUDA kernel to fuse gather -> matrix multiplication in PointConv which improves speed.
    Right now, it is numerically correct, but somehow it will mysteriously reduce training accuracy, hence only recommended to use during testing time
    '''

    def __init__(self):
        super(PConv, self).__init__()

    @staticmethod
    def forward(input_features, neighbor_inds, weightnet, additional_features=None):
        if additional_features is None:
            additional_features = torch.zeros(input_features.shape[0], input_features.shape[1], neighbor_inds.shape[2], 0)
        return PConvFunction.apply(
            input_features,
            neighbor_inds,
            weightnet,
            additional_features)


def VI_coordinate_transform(localized_xyz, gathered_norm, sparse_xyz_norm, K):
    """
    Compute the viewpoint-invariance aware relative position encoding in VI_PointConv
    From: X. Li et al. Improving the Robustness of Point Convolution on k-Nearest Neighbor Neighborhoods with a Viewpoint-Invariant Coordinate Transform. WACV 2023
    Code copyright 2020 Xingyi Li (MIT License)
    Input:
        dense_xyz: 3D coordinates (note VI only works on 3D)
        nei_inds: indices of neighborhood points for each point
        dense_xyz_norm: surface normals for each point
        sparse_xyz_norm: surface normals for each point in the lower resolution (normally
                the same as dense_xyz_norm, except when downsampling)
    Return:
        VI-transformed point coordinates: a concatenation of rotation+scale invariant dimensions, scale-invariant dimensions and non-invariant dimensions
    """
    r_hat = F.normalize(localized_xyz, dim=3)
    v_miu = sparse_xyz_norm.unsqueeze(
        dim=2) - torch.matmul(
        sparse_xyz_norm.unsqueeze(
            dim=2), r_hat.permute(
                0, 1, 3, 2)).permute(
                    0, 1, 3, 2) * r_hat
    v_miu = F.normalize(v_miu, dim=3)
    w_miu = torch.cross(r_hat, v_miu, dim=3)
    w_miu = F.normalize(w_miu, dim=3)
    theta1 = torch.matmul(gathered_norm, sparse_xyz_norm.unsqueeze(dim=3))
    theta2 = torch.matmul(r_hat, sparse_xyz_norm.unsqueeze(dim=3))
    theta3 = torch.sum(r_hat * gathered_norm, dim=3, keepdim=True)
    theta4 = torch.matmul(localized_xyz, sparse_xyz_norm.unsqueeze(dim=3))
    theta5 = torch.sum(gathered_norm * r_hat, dim=3, keepdim=True)
    theta6 = torch.sum(gathered_norm * v_miu, dim=3, keepdim=True)
    theta7 = torch.sum(gathered_norm * w_miu, dim=3, keepdim=True)
    theta8 = torch.sum(
        localized_xyz *
        torch.cross(
            gathered_norm,
            sparse_xyz_norm.unsqueeze(
                dim=2).repeat(
                1,
                1,
                K,
                1),
            dim=3),
        dim=3,
        keepdim=True)
    theta9 = torch.norm(localized_xyz, dim=3, keepdim=True)
    return torch.cat([theta1,
                      theta2,
                      theta3,
                      theta4,
                      theta5,
                      theta6,
                      theta7,
                      theta8,
                      theta9,
                      localized_xyz],
                     dim=3).contiguous()


# We did not like that the pyTorch batch normalization requires C to be the 2nd dimension of the Tensor
# It's hard to deal with it during training time, but we can fuse it during inference time
# This one takes in a 4D tensor of shape BNKC, run a linear layer and a BN layer, and then fuses it during inference time
# Output is BNKC as well
# B is batch size, N is number of points, K is number of neighbors
# one would need to call the fuse function during inference time (see
# utils.replace_bn_layers)
class Linear_BN(torch.nn.Module):
    def __init__(
            self,
            in_dim,
            out_dim,
            bn_ver='2d',
            bn_weight_init=1,
            bn_momentum=0.1):
        super().__init__()
        self.c = torch.nn.Linear(in_dim, out_dim)
        self.bn_ver = bn_ver
        if bn_ver == '2d':
            bn = CpBatchNorm2d(out_dim, momentum=bn_momentum)
        else:
            bn = torch.nn.BatchNorm1d(out_dim, momentum=bn_momentum)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
#        torch.nn.init.constant_(bn.bias, 0)
        self.bn = bn

    @torch.no_grad()
    @torch.jit.ignore()
    def fuse(self):
        w = self.bn.weight / (self.bn.running_var + self.bn.eps) ** 0.5
        w = self.c.weight * w[:, None]
        b = self.bn.bias + (self.c.bias - self.bn.running_mean) * self.bn.weight / \
            (self.bn.running_var + self.bn.eps)**0.5
        new_layer = torch.nn.Linear(w.size(1), w.size(0))
        new_layer.weight.data.copy_(w)
        new_layer.bias.data.copy_(b)
        return new_layer

    def forward(self, x):
        x = self.c(x)
        if self.bn_ver == '2d':
            return self.bn(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        else:
            return self.bn(x.permute(0, 2, 1)).permute(0, 2, 1)


# Linear_BN + Leaky ReLU activation
class UnaryBlock(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn, bn_momentum, no_relu=False):
        """
        Initialize a standard unary block with its ReLU and BatchNorm.
        :param in_dim: dimension input features
        :param out_dim: dimension input features
        :param use_bn: boolean indicating if we use Batch Norm
        :param bn_momentum: Batch norm momentum
        """

        super(UnaryBlock, self).__init__()
        self.bn_momentum = bn_momentum
        self.use_bn = use_bn
        self.no_relu = no_relu
        self.in_dim = in_dim
        self.out_dim = out_dim
        if use_bn:
            self.mlp = Linear_BN(
                in_dim,
                out_dim,
                bn_momentum=bn_momentum,
                bn_ver='1d')
        else:
            self.mlp = nn.Linear(in_dim, out_dim)
        if not no_relu:
            self.leaky_relu = nn.LeakyReLU(0.1)
        else:
            self.leaky_relu = nn.Identity()
        return

    def forward(self, x):
        x = self.mlp(x)
        if not self.no_relu:
            x = self.leaky_relu(x)
        return x

    def __repr__(self):
        return 'UnaryBlock(in_feat: {:d}, out_feat: {:d}, BN: {:s}, ReLU: {:s})'.format(
            self.in_dim, self.out_dim, str(self.use_bn), str(not self.no_relu))
