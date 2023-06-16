import os
import sys
from time import time
import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from timm.models.layers import DropPath
from easydict import EasyDict

from util.checkpoint import CheckpointFunction
from util.cp_batchnorm import CpBatchNorm2d

import pcf_cuda
#from torch.profiler import record_function, profile, ProfilerActivity
#from pytorch_memlab import LineProfiler, profile

def get_default_configs(cfg, num_level=5, base_dim=64):
    # Number of downsampling stages
    cfg.num_level = num_level
    # The dimensionality of the first stage
    cfg.base_dim = base_dim
    # Feature dimensionality for each stage
    if 'feat_dim' not in cfg.keys():
        cfg.feat_dim  = [base_dim * (i + 1) for i in range(cfg.num_level + 1)]
    # Whether to use the viewpoint-invariant coordinate transforms (Xingyi Li et al. WACV 2023)
    if 'USE_VI' not in cfg.keys():
        cfg.USE_VI = True
    # Whether to concatenate positional encoding into features
    if 'USE_PE' not in cfg.keys():
        cfg.USE_PE = True
    if 'transformer_type' not in cfg.keys():
        cfg.transformer_type = 'PCF'
    if 'attention_type' not in cfg.keys():
        cfg.attention_type = 'subtraction'
    # Whether to use a layer norm in the guidance computation
    if 'layer_norm_guidance' not in cfg.keys():
        cfg.layer_norm_guidance = False
    # Whether to use drop path
    if 'drop_path_rate' not in cfg.keys():
        cfg.drop_path_rate = 0.
    # Whether to use batch normalization
    if 'BATCH_NORM' not in cfg.keys():
        cfg.BATCH_NORM = True
    # Dropout rate
    if 'dropout_rate' not in cfg.keys():
        cfg.dropout_rate = 0.
    # Whether to time the individual components of the PointConv layers
    if 'TIME' not in cfg.keys():
        cfg.TIME = False
    # Whether to use coordinates as features too
    if 'USE_XYZ' not in cfg.keys():
        cfg.USE_XYZ = True
    # The dimensionality of the point cloud
    if 'point_dim' not in cfg.keys():
        cfg.point_dim = 3
    # Transformer_type: this can be either PCF or PointTransformer
    # for now, if it's not set to PCF then we will use PointTransformer
    if 'transformer_type' not in cfg.keys():
        cfg.transformer_type = 'PCF'
    if 'mid_dim_back' not in cfg.keys():
        cfg.mid_dim_back = 1
    if 'use_level_1' not in cfg.keys():
        cfg.use_level_1 = True
    # Whether to use the CUDA kernels for PointConv and PointConvFormer which reduces 
    # gather operations and save a few milliseconds and some memory
    if 'USE_CUDA_KERNEL' not in cfg.keys():
        cfg.USE_CUDA_KERNEL = False
    return cfg


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
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

class PCFFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_feat, neighbor_inds, guidance, weightnet):
        # Make sure we are not computing gradient on neighbor_inds
        neighbor_inds.requires_grad = False
        output = pcf_cuda.pcf_forward(input_feat, neighbor_inds, guidance, weightnet)
        ctx.save_for_backward(input_feat, neighbor_inds, guidance, weightnet)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input, grad_guidance, grad_weight = pcf_cuda.pcf_backward(grad_output.contiguous(), *ctx.saved_tensors)
        return grad_input, None, grad_guidance, grad_weight


class PCF(torch.nn.Module):
    def __init__(self):
        super(PCF, self).__init__()
    def forward(input_features, neighbor_inds, guidance, weightnet):
        return PCFFunction.apply(input_features, neighbor_inds, guidance, weightnet)

class PConvFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_feat, neighbor_inds, weightnet, additional_features):
        # Make sure we are not computing gradient on neighbor_inds
        neighbor_inds.requires_grad = False
        output = pcf_cuda.pconv_forward(input_feat, neighbor_inds, weightnet, additional_features)
        ctx.save_for_backward(input_feat, neighbor_inds, weightnet, additional_features)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input, grad_weight, grad_additional = pcf_cuda.pconv_backward(grad_output.contiguous(), *ctx.saved_tensors)
        return grad_input, None, grad_weight, grad_additional

class PConv(torch.nn.Module):
    def __init__(self):
        super(PConv, self).__init__()
    def forward(input_features, neighbor_inds, weightnet, additional_features):
        return PConvFunction.apply(input_features, neighbor_inds, weightnet, additional_features)

def VI_coordinate_transform(localized_xyz, gathered_norm, sparse_xyz_norm, K):
    """

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
    v_miu = sparse_xyz_norm.unsqueeze(dim=2) - torch.matmul(sparse_xyz_norm.unsqueeze(dim=2), r_hat.permute(0, 1, 3, 2)).permute(0, 1, 3, 2) * r_hat
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
    theta8 = torch.sum(localized_xyz * torch.cross(gathered_norm, sparse_xyz_norm.unsqueeze(dim=2).repeat(1, 1, K, 1), dim=3), dim=3, keepdim=True)
    theta9 = torch.norm(localized_xyz, dim=3, keepdim=True)
    return torch.cat(
            [theta1, theta2, theta3, theta4, theta5, theta6, theta7, theta8, theta9, localized_xyz], dim=3).contiguous()

# We did not like that the pyTorch batch normalization requires C to be the 2nd dimension of the Tensor
# It's hard to deal with it during training time, but we can fuse it during inference time
# This one takes in a 4D tensor of shape BNKC, run a linear layer and a BN layer, and then fuses it during inference time
# Output is BNKC as well
# B is batch size, N is number of points, K is number of neighbors
# one would need to call the fuse function during inference time (see utils.replace_bn_layers)
class Linear_BN(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bn_ver = '2d', bn_weight_init=1, bn_momentum = 0.1):
        super().__init__()
        self.c =  torch.nn.Linear(in_dim, out_dim)
        self.bn_ver = bn_ver
        if bn_ver == '2d':
            bn = CpBatchNorm2d(out_dim, momentum = bn_momentum)
        else:
            bn = torch.nn.BatchNorm1d(out_dim, momentum = bn_momentum)
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
            return self.bn(x.permute(0,3,2,1)).permute(0,3,2,1)
        else:
            return self.bn(x.permute(0,2,1)).permute(0,2,1)

# Multi-head Guidance:
# Input: guidance_query: input features (B x N x K x C)
#        guidance_key: also input features (but less features when downsampling)
#        pos_encoding: if not None, then position encoding is concatenated with the features
# Output: guidance_features: (B x N x K x num_heads)

class MultiHeadGuidanceNewV1(nn.Module):
    """ Multi-head guidance to increase model expressivitiy"""
    def __init__(self, cfg, num_heads: int, num_hiddens: int):
        super().__init__()
        # assert num_hiddens % num_heads == 0, 'num_hiddens: %d, num_heads: %d'%(num_hiddens, num_heads)
        self.cfg = cfg
        self.dim = num_hiddens
        self.num_heads = num_heads

        self.layer_norm_q = nn.LayerNorm(num_hiddens) if cfg.layer_norm_guidance else nn.Identity()
        self.layer_norm_k = nn.LayerNorm(num_hiddens) if cfg.layer_norm_guidance else nn.Identity()

        self.mlp = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        mlp_dim = [self.dim, 8, num_heads]
        for ch_in, ch_out in zip(mlp_dim[:-2], mlp_dim[1:-1]):
            if cfg.BATCH_NORM:
                self.mlp.append(Linear_BN(ch_in, ch_out))
            else:
            	self.mlp.append(nn.Linear(ch_in, ch_out))
        self.mlp.append(nn.Linear(mlp_dim[-2], mlp_dim[-1]))
 #           if cfg.BATCH_NORM:
 #               self.mlp_bns.append(nn.BatchNorm1d(ch_out))

        # self.pos_encoding = nn.Linear(self.dim, num_heads)

    def forward(self, guidance_query, guidance_key):#, pos_encoding=None):

        # import ipdb; ipdb.set_trace()
        # attention bxnxkxc
        batch_dim,  n, k, _ = guidance_query.shape

#        guidance_query = guidance_query.view(batch_dim, self.dim, k, n).permute(0, 2, 3, 1)
#        guidance_key = guidance_key.view(batch_dim, self.dim, k, n).permute(0, 2, 3, 1)

        # scores = torch.einsum('bdhkn, bdhkn->bhkn', guidance_query, guidance_key) / self.dim**.5
        # scores = scores.unsqueeze(-1)
        
        scores = self.layer_norm_q(guidance_query) - self.layer_norm_k(guidance_key)
#        scores = scores if pos_encoding is None else scores + pos_encoding.permute(0, 2, 3, 1)

        for i, layer in enumerate(self.mlp):
            scores = layer(scores)

            if i == len(self.mlp) - 1:
                # pos_encoding = self.pos_encoding(pos_encoding.permute(0, 2, 3, 1))
                # scores = torch.sigmoid(scores + pos_encoding)
                scores = torch.sigmoid(scores)
                # scores = torch.sigmoid(scores)
                #scores = torch.nn.functional.softmax(scores, dim = 2)
                #scores = F.relu(scores)
                # scores = torch.tanh(scores).squeeze(-1)
            else:
                scores = F.relu(scores, inplace=True)

        return scores

class MultiHeadGuidanceQK(nn.Module):
    """ Multi-head guidance to increase model expressivitiy"""
    def __init__(self, cfg, num_heads: int, num_hiddens: int, key_dim: int):
        super().__init__()
        assert num_hiddens % num_heads == 0, 'num_hiddens: %d, num_heads: %d'%(num_hiddens, num_heads)
        self.cfg = cfg
        self.dim = num_hiddens
        self.num_heads = num_heads
        self.key_dim = key_dim
        #self.key_dim = key_dim = num_hiddens // num_heads
        self.scale = self.key_dim ** -0.5
        self.qk_linear = Linear_BN(self.dim, key_dim * num_heads)
#        self.v_linear = nn.Linear(key_dim, int(self.dim / num_heads))


    def forward(self, q, k):
        # input q: b, n,k, c
        #       k: b, n,k,c

        # import ipdb; ipdb.set_trace()
        # compute q, k
        B, N, K, C = q.shape

        q = self.qk_linear(q)
        k = self.qk_linear(k)
        q = q.view(B, N, K, self.num_heads,-1)
        k = k.view(B, N, K, self.num_heads,-1)
        #actually there is only one center..
        k = k[:,:,:1,:,:]
        q = q.transpose(2,3)
        k = k.permute(0,1,3,4,2)

        # compute attention 
        attn_score = (q@k) * self.scale
        attn_score = attn_score[:,:,:,:,0].transpose(2,3)
        # attn_score = attn_score.sigmoid() # b, n, nei_size, nh, 1
        #attn_score = F.softmax(attn_score, dim = 2)
        attn_score = torch.sigmoid(attn_score)

        # prepare v 
#        gathered_v = index_points(v.flatten(2, 3), nei_inds).view(B, N, nei_size, self.num_heads, -1)
#        attned_v = self.v_linear(gathered_v * attn_score).view(B, N, nei_size, self.dim)

        return attn_score
        
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
            self.mlp = Linear_BN(in_dim, out_dim, bn_momentum = bn_momentum, bn_ver = '1d')
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
        return 'UnaryBlock(in_feat: {:d}, out_feat: {:d}, BN: {:s}, ReLU: {:s})'.format(self.in_dim,
                                                                                        self.out_dim,
                                                                                        str(self.use_bn),
                                                                                        str(not self.no_relu))


def _bn_function_factory(mlp_convs):
    def bn_function(*inputs):
        output = inputs[0]
        for i, conv in enumerate(mlp_convs):
            output = F.relu(conv(output), inplace= True)
        return output
    return bn_function

# WeightNet for PointConv: Input: Coordinates for all the kNN neighborhoods
#                          input shape is B x N x K x in_channel, B is batch size, in_channel is the dimensionality of 
#                          the coordinates (usually 3 for 3D or 2 for 2D, 12 for VI), K is the neighborhood size,
#                          N is the number of points
#                          This layer runs 2 MLP layers on the point coordinates and outputs
#                          generated weights for each neighbor of each point. The weights will
#                          then be matrix-multiplied with the input to perform convolution
#                          Output: The generated weights B x N x K x C_mid
#                          
class WeightNet(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_unit=[8, 8], efficient = False):
        super(WeightNet, self).__init__()

        self.mlp_convs = nn.ModuleList()
        self.efficient = efficient
        if hidden_unit is None or len(hidden_unit) == 0:
            self.mlp_convs.append(Linear_BN(in_channel, out_channel))
        else:
            self.mlp_convs.append(Linear_BN(in_channel, hidden_unit[0]))
            for i in range(1, len(hidden_unit)):
                self.mlp_convs.append(Linear_BN(hidden_unit[i-1], hidden_unit[i]))
            self.mlp_convs.append(Linear_BN(hidden_unit[-1], out_channel))

    def real_forward(self, localized_xyz):
        # xyz : BxNxKxC
        weights = localized_xyz
        for i, conv in enumerate(self.mlp_convs):
            weights = conv(weights)
#        if i < len(self.mlp_convs) - 1:
            weights = F.relu(weights, inplace=True)

        return weights
    def forward(self, localized_xyz):
        if self.efficient and self.training:
            # Try this so that weights have gradient
#            weights = self.mlp_convs[0](localized_xyz)
            conv_bn_relu = _bn_function_factory(self.mlp_convs)
            self.dummy = torch.zeros(1, dtype = torch.float32, requires_grad = True, device = localized_xyz.device)
            args = [localized_xyz + self.dummy]
            if self.training:
                for i, conv in enumerate(self.mlp_convs):
                     args += tuple(conv.bn.parameters())
                     args += tuple(conv.c.parameters())
                weights = CheckpointFunction.apply(conv_bn_relu, 1, *args)
#            weights = checkpoint.checkpoint(self.checkpoint_forward, localized_xyz, self.dummy_var)
        else:
            weights = self.real_forward(localized_xyz)
        return weights

# PointConvFormer main layer
class GuidedConvStridePENewAfter(nn.Module):
    def __init__(self, in_channel, out_channel, cfg, weightnet=[9, 16], num_heads=4, guidance_feat_len=32):
        super(GuidedConvStridePENewAfter, self).__init__()
        self.cfg = cfg
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.num_heads = num_heads

        self.drop_path = DropPath(cfg.drop_path_rate) if cfg.drop_path_rate > 0. else nn.Identity()

        if cfg.BATCH_NORM:
            self.mlp_conv = Linear_BN(12, guidance_feat_len)
        else:
            self.mlp_conv = nn.Linear(12, guidance_feat_len)

        # First downscaling mlp
        if in_channel != out_channel // 4:
            self.unary1 = UnaryBlock(in_channel, out_channel // 4, use_bn=True, bn_momentum=0.1)
        else:
            self.unary1 = nn.Identity()
            
        self.guidance_unary = UnaryBlock(out_channel // 4, guidance_feat_len, use_bn=True, bn_momentum=0.1, no_relu=True)

        # check last_ch % num_heads == 0
        assert (out_channel // 2) % num_heads == 0
        if cfg.attention_type == 'subtraction':
            self.guidance_weight = MultiHeadGuidanceNewV1(cfg, num_heads, 2 * guidance_feat_len)
        else:
            self.guidance_weight = MultiHeadGuidanceQK(cfg, num_heads, 2 * guidance_feat_len, key_dim  = 16)
#        self.mix_linear = nn.Conv2d(out_channel // 2, out_channel // 2, 1)
#        self.mix_linear = nn.Linear(out_channel // 4 + guidance_feat_len, out_channel // 4)

        self.weightnet = WeightNet(weightnet[0], weightnet[1], efficient = True)
        if cfg.BATCH_NORM:
            self.linear = Linear_BN(out_channel // 4 * weightnet[-1], out_channel // 2, bn_ver = '1d')
        else:
            self.linear = nn.Linear(out_channel // 4 * weightnet[-1], out_channel // 2)

        self.dropout = nn.Dropout(p=cfg.dropout_rate) if cfg.dropout_rate > 0. else nn.Identity()

        # Second upscaling mlp
        self.unary2 = UnaryBlock(out_channel // 2, out_channel, use_bn=True, bn_momentum=0.1, no_relu=True)

        # Shortcut optional mpl
        if in_channel != out_channel:
            self.unary_shortcut = UnaryBlock(in_channel, out_channel, use_bn=True, bn_momentum=0.1, no_relu=True)
        else:
            self.unary_shortcut = nn.Identity()

        # Other operations
        self.leaky_relu = nn.LeakyReLU(0.1)

        return   
#    @profile
    def forward(self, dense_xyz, dense_feats, nei_inds, dense_xyz_norm, sparse_xyz = None, sparse_xyz_norm = None, vi_features = None):
        """
        dense_xyz: tensor (batch_size, num_points, 3)
        sparse_xyz: tensor (batch_size, num_points2, 3)
        dense_feats: tensor (batch_size, num_points, num_dims)
        nei_inds: tensor (batch_size, num_points2, K)
        """
        B, N, D = dense_xyz.shape
        if sparse_xyz is not None:
            _, M, _ = sparse_xyz.shape
        else:
            M = N
        _, _, in_ch = dense_feats.shape
        _, _, K = nei_inds.shape
        if self.cfg.TIME:
             print('')
             print('***** Time Breakdown of GuidedConvStridePENewAfter *****')
             torch.cuda.synchronize()
             t0 = time()
        # first downscaling mlp
        feats_x = self.unary1(dense_feats)
        guidance_feat = self.guidance_unary(feats_x)
        if self.cfg.TIME:
             torch.cuda.synchronize()
             t1 = time()
             print('initial unary time: ', t1-t0)

        if self.cfg.TIME:
            torch.cuda.synchronize()
            t1 = time()
            print('initial unary time: ', t1-t0)

        gathered_xyz = index_points(dense_xyz, nei_inds)
        # localized_xyz = gathered_xyz - sparse_xyz.view(B, M, 1, D) #[B, M, K, D]
        if sparse_xyz is not None:
        	localized_xyz = gathered_xyz - sparse_xyz.unsqueeze(dim=2)
        else:
        	localized_xyz = gathered_xyz - dense_xyz.unsqueeze(dim=2)
        gathered_norm = index_points(dense_xyz_norm, nei_inds)
        
        if self.cfg.TIME:
            torch.cuda.synchronize()
            t2 = time()
            print('gather xyz time: ', t2-t1)

        if self.cfg.USE_VI is True:
            if vi_features is None:
              if sparse_xyz is not None:
                weightNetInput = VI_coordinate_transform(localized_xyz, gathered_norm, sparse_xyz_norm, K)
              else:
                weightNetInput = VI_coordinate_transform(localized_xyz, gathered_norm, dense_xyz_norm, K)

            else:
              weightNetInput = vi_features
        else:
            weightNetInput = localized_xyz
        # Encode weightNetInput to be higher dimensional to match with gathered feat
        feat_pe = self.mlp_conv(weightNetInput)
        feat_pe = F.relu(feat_pe)

        if self.cfg.TIME:
            torch.cuda.synchronize()
            t3 = time()
            print('weightnet time: ', t3-t2)
        
        if self.cfg.TIME:
             torch.cuda.synchronize()
             t3 = time()
             print('guidance weightnet time: ', t3-t2)
        gathered_feat = index_points(feats_x, nei_inds)
        # First downsample on the feature dimension, so that it matches the position encoding dimension
        guidance_x = self.guidance_unary(feats_x)
        # Gather features on this low dimensionality is faster and uses less memory
        gathered_feat2 = index_points(guidance_x, nei_inds)  # [B, M, K, in_ch]
        # new_feat = gathered_feat.permute(0, 3, 2, 1)
        guidance_feature = torch.cat([gathered_feat2, feat_pe], dim=-1)

        if self.cfg.TIME:
             torch.cuda.synchronize()
             t4 = time()
             print('Gather feature time: ', t4-t3)
        
        if self.cfg.TIME:
            torch.cuda.synchronize()
            t4 = time()
            print('Gather feature time: ', t4-t3)
        # import ipdb; ipdb.set_trace()
        # guidance_feature = new_feat
        guidance_query = guidance_feature # b m k c
#        if M == N:
#            guidance_key = guidance_feature[:,:,:1,:].repeat(1,1,K,1)
#        else:
        guidance_key = guidance_feature.max(dim=2, keepdim=True)[0].repeat(1, 1, K, 1)
        guidance_score = self.guidance_weight(guidance_query, guidance_key) # b n k num_heads
        
        if self.cfg.TIME:
             torch.cuda.synchronize()
             t45 = time()
             print('Guidance main computation time: ', t45- t4)
             t4 = t45

        # apply guide to features
#        if self.cfg.USE_PE:
#            gathered_feat = self.mix_linear(torch.cat([gathered_feat, feat_pe],dim=-1))#self.mix_linear(new_feat)
        gathered_feat = gathered_feat.permute(0,3,2,1)
        gathered_feat = (gathered_feat.view(B, -1, self.num_heads, K, M) * guidance_score.permute(0,3,2,1)).view(B, -1, K, M)

        weights = self.weightnet(weightNetInput)

        if self.cfg.TIME:
            torch.cuda.synchronize()
            t5 = time()
            print('WeightNet time: ', t5-t4)

        # localized_xyz = localized_xyz.permute(0, 3, 2, 1)
        # weights = self.weightnet(localized_xyz)*nei_inds_mask.permute(0,2,1).unsqueeze(dim=1)

        if self.cfg.TIME:
            torch.cuda.synchronize()
            t5 = time()
            print('Prepare guided feature time: ', t5-t4)
        gathered_feat = torch.matmul(input=gathered_feat.permute(0,3,1,2).contiguous(), other=weights).view(B, M, -1)
        if self.cfg.TIME:
            torch.cuda.synchronize()
            t6 = time()
            print('Matmul time: ', t6-t5)
        # new_feat = new_feat/nn_idx_divider.unsqueeze(dim=-1)

#        new_feat = new_feat.view(B, M, -1)
        new_feat = self.linear(gathered_feat)
        new_feat = F.relu(new_feat, inplace=True)

        # Dropout
        new_feat = self.dropout(new_feat)

        # Second upscaling mlp
        new_feat = self.unary2(new_feat)

        # TODO: some speed-up opportunities here to shave a few milliseconds
        if sparse_xyz is not None:
            sparse_feats = torch.max(index_points(dense_feats, nei_inds), dim = 2)[0]
        else:
        	sparse_feats = dense_feats

        shortcut = self.unary_shortcut(sparse_feats)

        new_feat = self.leaky_relu(self.drop_path(new_feat) + shortcut)
        if self.cfg.TIME:
            torch.cuda.synchronize()
            t7 = time()
            print('Final time: ', t7-t6)
            print('Total time: ', t7-t0)
            print('')

        return new_feat, weightNetInput
        
class PointTransformerLayer(nn.Module):
    def __init__(self, in_planes, out_planes, share_planes=8):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)
        self.linear_p = nn.Sequential(Linear_BN(3, 3, bn_ver='1d'), nn.ReLU(inplace=True), nn.Linear(3, out_planes))
        self.bn_w = nn.BatchNorm1d(mid_planes)
        self.linear_w = nn.Sequential(nn.ReLU(inplace=True),
                                    Linear_BN(mid_planes, mid_planes // share_planes, bn_ver='1d'),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(mid_planes // share_planes, out_planes // share_planes))
        self.softmax = nn.Softmax(dim=1)
        if in_planes != out_planes:
            self.unary_shortcut = UnaryBlock(in_planes, out_planes, use_bn=True, bn_momentum=0.1, no_relu=True)
        else:
            self.unary_shortcut = nn.Identity()
        self.leaky_relu = nn.LeakyReLU(0.1)
        
    def forward(self, xyz, feats, nei_ind, sparse_xyz = None,norm=None) -> torch.Tensor:
        #xyz: b, n, 3  nei_ind: b, n, k
        #feats: b, n, c
        # import ipdb; ipdb.set_trace()
        b, n, c_in = feats.shape
        _, _, k = nei_ind.shape
        if sparse_xyz is not None:
            _, M, _ = sparse_xyz.shape
        else:
            M = n

        feats_q, feats_k, feats_v = self.linear_q(feats), self.linear_k(feats), self.linear_v(feats)
#        feats_q = feats_q.squeeze(0)
        feats_k = index_points(feats_k, nei_ind).squeeze(0) # n, k, c_mid
        feats_v = index_points(feats_v, nei_ind).squeeze(0) # n, k, c_mid
        if sparse_xyz is not None:
            dxyz = (index_points(xyz,nei_ind) - sparse_xyz.unsqueeze(dim=2))
            feats_q = index_points(feats_q, nei_ind[:,:,0].unsqueeze(dim=2))
        else:
            dxyz = (index_points(xyz, nei_ind) - xyz.unsqueeze(dim=2)) # n, k, 3
            feats_q = feats_q.unsqueeze(dim=2)
        dxyz = dxyz.squeeze(0)
        for i, layer in enumerate(self.linear_p):
            dxyz = layer(dxyz)
        #print(feats_k.shape)
        #print(feats_q.shape)
        w = feats_k - feats_q[0] + dxyz.view(M, k, self.out_planes // self.mid_planes, self.mid_planes).sum(2) # n, k, c_mid
        w = w.transpose(1,2)
        w = self.bn_w(w)
        w = w.transpose(2,1)
        #print(w.shape)
        for i, layer in enumerate(self.linear_w): 
            w = layer(w)
        w = self.softmax(w)
        c = feats_v.shape[-1]; s = self.share_planes
        new_feats = ((feats_v + dxyz).view(M, k, s, c // s) * w.unsqueeze(2)).sum(1).view(M, c)
        if sparse_xyz is not None:
            sparse_feats = torch.max(index_points(feats, nei_ind), dim=2)[0]
        else:
            sparse_feats = feats
        shortcut = self.unary_shortcut(sparse_feats)
        new_feats = self.leaky_relu(new_feats + shortcut)
        return new_feats
        
# PointConv layer with a positional embedding concatenated to the features
class PointConvStridePE(nn.Module):
    def __init__(self, in_channel, out_channel, cfg, weightnet=[9, 16]):
        super(PointConvStridePE, self).__init__()
        self.cfg = cfg
        self.in_channel = in_channel
        self.out_channel = out_channel
        
        #print(cfg)

        self.drop_path = DropPath(cfg.drop_path_rate) if cfg.drop_path_rate > 0. else nn.Identity()

        # positonal encoder
        self.pe_convs = WeightNet(3, min(out_channel//4, 32), hidden_unit = [out_channel//4], efficient = True)
        last_ch = min(out_channel//4, 32)
#        self.mlp_convs = nn.ModuleList()
#        last_ch = 3
#        for out_ch in [out_channel//4, min(out_channel//4,32)]:
#            if cfg.BATCH_NORM:
#                self.mlp_convs.append(Linear_BN(last_ch, out_ch))
#            else:
#                self.mlp_convs.append(nn.Lienar(last_ch, out_ch))
#            last_ch = out_ch

        # First downscaling mlp
        if in_channel != out_channel // 4:
            self.unary1 = UnaryBlock(in_channel, out_channel // 4, use_bn=True, bn_momentum=0.1)
        else:
            self.unary1 = nn.Identity()

        self.weightnet = WeightNet(weightnet[0], weightnet[1], efficient = True)
        if cfg.BATCH_NORM:
            self.linear = Linear_BN((out_channel // 4 + last_ch) * weightnet[-1], out_channel // 2, bn_ver = '1d')
        else:
            self.linear = nn.Linear((out_channel // 4 + last_ch) * weightnet[-1], out_channel // 2)

        self.dropout = nn.Dropout(p=cfg.dropout_rate) if cfg.dropout_rate > 0. else nn.Identity()

        # Second upscaling mlp
        self.unary2 = UnaryBlock(out_channel // 2, out_channel, use_bn=True, bn_momentum=0.1, no_relu=True)

        # Shortcut optional mpl
        if in_channel != out_channel:
            self.unary_shortcut = UnaryBlock(in_channel, out_channel, use_bn=True, bn_momentum=0.1, no_relu=True)
        else:
            self.unary_shortcut = nn.Identity()

        # Other operations
        self.leaky_relu = nn.LeakyReLU(0.1)

        return
#    @profile
    def forward(self, dense_xyz, dense_feats, nei_inds, dense_xyz_norm, sparse_xyz = None, sparse_xyz_norm = None, vi_features = None):
        """
        dense_xyz: tensor (batch_size, num_points, 3)
        sparse_xyz: tensor (batch_size, num_points2, 3), if None, then assume sparse_xyz = dense_xyz
        dense_feats: tensor (batch_size, num_points, num_dims)
        nei_inds: tensor (batch_size, num_points2, K)
        """
        if self.cfg.TIME:
            print("time cut down of PointConvStridePE ************")
        if True:
        #with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA], record_shapes=True) as prof:
         B, N, D = dense_xyz.shape
         if sparse_xyz is not None:
            _, M, _ = sparse_xyz.shape
         else:
            M = N
         _, _, in_ch = dense_feats.shape
         _, _, K = nei_inds.shape

        # nei_inds = nei_inds.clone().detach()
        # nei_inds_mask = (nei_inds != -1).float()
        # nn_idx_divider = nei_inds_mask.sum(dim = -1)
        # nn_idx_divider[nn_idx_divider == 0] = 1
        # nei_inds[nei_inds == -1] = 0
        if self.cfg.TIME:
            torch.cuda.synchronize()
            t0 = time()
        # First downscaling mlp
        feats_x = self.unary1(dense_feats)

        if self.cfg.TIME: 
            torch.cuda.synchronize()
            t1 = time()
            print("unary1 time: ", t1 - t0)
        gathered_xyz = index_points(dense_xyz, nei_inds)
        # localized_xyz = gathered_xyz - sparse_xyz.view(B, M, 1, D) #[B, M, K, D]
        if sparse_xyz is not None:
            localized_xyz = gathered_xyz - sparse_xyz.unsqueeze(dim=2)
        else:
            localized_xyz = gathered_xyz - dense_xyz.unsqueeze(dim=2)
        gathered_norm = index_points(dense_xyz_norm, nei_inds)

        if self.cfg.TIME:
            torch.cuda.synchronize()
            t2 = time()
            print("indexing time: ", t2 - t1)
        feat_pe = self.pe_convs(localized_xyz)  # [B, M, K, D]
#        for i, conv in enumerate(self.mlp_convs):
#            feat_pe = conv(feat_pe)
#            feat_pe = F.relu(feat_pe, inplace=True)


        if self.cfg.TIME:         
            torch.cuda.synchronize()
            t3 = time()

            print("pe time: ", t3 - t2)
        if self.cfg.USE_VI is True:
          if vi_features is None:
              if sparse_xyz is not None:
                weightNetInput = VI_coordinate_transform(localized_xyz, gathered_norm, sparse_xyz_norm, K)
              else:
                weightNetInput = VI_coordinate_transform(localized_xyz, gathered_norm, dense_xyz_norm, K)
          else:
              weightNetInput = vi_features
        else:
            weightNetInput = localized_xyz

        if self.cfg.TIME:
            torch.cuda.synchronize()
            t4 = time()
            print("VI Features time: ", t4 - t3)

        # If not using CUDA kernel, then we need to sparse gather the features here
        if not self.cfg.USE_CUDA_KERNEL:
            gathered_feat = index_points(feats_x, nei_inds)  # [B, M, K, in_ch]
            new_feat = torch.cat([gathered_feat, feat_pe], dim=-1)
        # new_feat = gathered_feat.permute(0, 3, 2, 1)

        if self.cfg.TIME:
           torch.cuda.synchronize()
           t5 = time()
           print("Gather feature time: ", t5-t4)
           t4 = t5

        weights = self.weightnet(weightNetInput)

        if self.cfg.TIME:
            torch.cuda.synchronize()
            t5 = time()
            print("weightnet time: ", t5 - t4)

        if self.cfg.USE_CUDA_KERNEL:
            feats_x = feats_x.contiguous()
            nei_inds = nei_inds.contiguous()
            weights = weights.contiguous()
            feat_pe = feat_pe.contiguous()
            new_feat = PConv.forward(feats_x, nei_inds, weights, feat_pe)
        else:
#        print(new_feat.shape)
            new_feat = torch.matmul(input=new_feat.permute(0, 1,3, 2), other=weights).view(B, M, -1)
        # new_feat = new_feat/nn_idx_divider.unsqueeze(dim=-1)

        new_feat = self.linear(new_feat)
        new_feat = F.relu(new_feat, inplace=True)

        # Dropout
        new_feat = self.dropout(new_feat)


        # Second upscaling mlp
        new_feat = self.unary2(new_feat)
        if sparse_xyz is not None:
          sparse_feats = torch.max(index_points(dense_feats, nei_inds), dim = 2)[0]
        else:
          sparse_feats = dense_feats

        shortcut = self.unary_shortcut(sparse_feats)

        new_feat = self.leaky_relu(self.drop_path(new_feat) + shortcut)

        if self.cfg.TIME:
            torch.cuda.synchronize()
            t6 = time()
            print("pointconv[matmul, linear,...] time: ", t6 - t5)

        return new_feat, weightNetInput

# This layer implements VI_PointConv and PointConv (set USE_VI = false)
class PointConv(nn.Module):
    def __init__(self, in_channel, out_channel, mlp, cfg, weightnet=[9, 16], USE_VI = None):
        super(PointConv, self).__init__()
        self.cfg = cfg
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.USE_VI = cfg.USE_VI
        if USE_VI is not None:
            self.USE_VI = USE_VI
        last_ch = in_channel
        if cfg.USE_PE:
            if self.USE_VI:
              last_ch = in_channel + 12
            else:
              last_ch = in_channel + 3
            if mlp:
              for out_ch in mlp:
                if cfg.BATCH_NORM:
                    self.mlp_convs.append(Linear_BN(last_ch, out_ch))
                else:
                    self.mlp_convs.append(nn.Linear(last_ch, out_ch))
                last_ch = out_ch
        else:
            last_ch = in_channel
        self.weightnet = WeightNet(weightnet[0], weightnet[1], efficient = True)
        if cfg.BATCH_NORM:
            self.linear = Linear_BN(last_ch * weightnet[-1], out_channel, bn_ver = '1d')
        else:
            self.linear = nn.Linear(last_ch * weightnet[-1], out_channel)

        self.dropout = nn.Dropout(p=cfg.dropout_rate) if cfg.dropout_rate > 0. else nn.Identity()

    def forward(self, dense_xyz, dense_feats, nei_inds, dense_xyz_norm = None, sparse_xyz = None, sparse_xyz_norm = None):
        """
        dense_xyz: tensor (batch_size, num_points, 3)
        sparse_xyz: tensor (batch_size, num_points2, 3)
        dense_feats: tensor (batch_size, num_points, num_dims)
        nei_inds: tensor (batch_size, num_points2, K)
        dense_xyz_norm: normals of the dense xyz, tensor (batch_size, num_points, 3)
        sparse_xyz_norm: normals of the sparse xyz, tensor (batch_size, num_points2, 3)
        norms are required if USE_VI is true
        """
        B, N, D = dense_xyz.shape
        if sparse_xyz is not None:
            _, M, _ = sparse_xyz.shape
        else:
            M = N
        _, _, in_ch = dense_feats.shape
        _, _, K = nei_inds.shape

        # nei_inds = nei_inds.clone().detach()
        # nei_inds_mask = (nei_inds != -1).float()
        # nn_idx_divider = nei_inds_mask.sum(dim = -1)
        # nn_idx_divider[nn_idx_divider == 0] = 1
        # nei_inds[nei_inds == -1] = 0

        gathered_xyz = index_points(dense_xyz, nei_inds)
        # localized_xyz = gathered_xyz - sparse_xyz.view(B, M, 1, D) #[B, M, K, D]
        if sparse_xyz is not None:
            localized_xyz = gathered_xyz - sparse_xyz.unsqueeze(dim=2)
        else:
            localized_xyz = gathered_xyz - dense_xyz.unsqueeze(dim=2)

        if self.USE_VI is True:
          gathered_norm = index_points(dense_xyz_norm, nei_inds)
          if sparse_xyz is not None:
              weightNetInput = VI_coordinate_transform(localized_xyz, gathered_norm, sparse_xyz_norm, K)
          else:
        	  weightNetInput = VI_coordinate_transform(localized_xyz, gathered_norm, dense_xyz_norm, K)
        else:
            weightNetInput = localized_xyz


        gathered_feat = index_points(dense_feats, nei_inds)  # [B, M, K, in_ch]
        if self.cfg.USE_PE:
            gathered_feat = torch.cat([gathered_feat, weightNetInput], dim=-1)

            for i, conv in enumerate(self.mlp_convs):
                gathered_feat = F.relu(conv(gathered_feat), inplace=True)
        
        weights = self.weightnet(weightNetInput)

        # localized_xyz = localized_xyz.permute(0, 3, 2, 1)
        # weights = self.weightnet(localized_xyz)*nei_inds_mask.permute(0,2,1).unsqueeze(dim=1)
        new_feat = torch.matmul(input=gathered_feat.permute(0, 1, 3, 2), other=weights).view(B, M, -1)
        # new_feat = new_feat/nn_idx_divider.unsqueeze(dim=-1)
        new_feat = F.relu(self.linear(new_feat), inplace=True)

        # Dropout
        new_feat = self.dropout(new_feat)

        return new_feat, weightNetInput


# PointConvTranspose (upsampling) layer
# one needs to input dense_xyz (high resolution point coordinates after upsampling) and sparse_xyz (low-resolution)
# and this layer would put features to the points at dense_xyz
class PointConvTransposePE(nn.Module):
    def __init__(self, in_channel, out_channel, cfg, weightnet=[9, 16], mlp2=None):
        super(PointConvTransposePE, self).__init__()
        self.cfg = cfg
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.drop_path = DropPath(cfg.drop_path_rate) if cfg.drop_path_rate > 0. else nn.Identity()
        self.unary1 = UnaryBlock(in_channel, out_channel, use_bn=True, bn_momentum=0.1)

        # positonal encoder
        self.pe_convs = nn.ModuleList()
        if cfg.USE_PE:
            self.pe_convs = WeightNet(3, min(out_channel//4,32), hidden_unit = [out_channel//4], efficient = True)
            last_ch = min(out_channel//4,32)
        else:
            self.pe_convs = nn.ModuleList()
            last_ch = 0
 
        self.weightnet = WeightNet(weightnet[0], weightnet[1], efficient = True)
        if cfg.BATCH_NORM:
            self.linear = Linear_BN((last_ch + out_channel) * weightnet[-1], out_channel, bn_ver = '1d')
        else:
            self.linear = nn.Linear((last_ch + out_channel) * weightnet[-1], out_channel)

        self.dropout = nn.Dropout(p=cfg.dropout_rate) if cfg.dropout_rate > 0. else nn.Identity()

        self.mlp2_convs = nn.ModuleList()
        self.mlp2_bns = nn.ModuleList()
        if mlp2 is not None:
            for i in range(1, len(mlp2)):
                if cfg.BATCH_NORM:
                    self.mlp2_convs.append(Linear_BN(mlp2[i-1],mlp2[i], bn_ver = '1d'))
                else:
                    self.mlp2_convs.append(nn.Linear(mlp2[i-1],mlp2[i]))
#    @profile
    def forward(self, sparse_xyz, sparse_feats, nei_inds, sparse_xyz_norm,  dense_xyz, dense_feats,dense_xyz_norm, vi_features = None):
        """
        dense_xyz: tensor (batch_size, num_points, 3)
        sparse_xyz: tensor (batch_size, num_points2, 3)
        dense_feats: tensor (batch_size, num_points, num_dims)
        nei_inds: tensor (batch_size, num_points2, K)
        """
        B, N, D = sparse_xyz.shape
        _, M, _ = dense_xyz.shape
        _, _, in_ch = sparse_feats.shape
        _, _, K = nei_inds.shape

        # nei_inds = nei_inds.clone().detach()
        # nei_inds_mask = (nei_inds != -1).float()
        # nn_idx_divider = nei_inds_mask.sum(dim = -1)
        # nn_idx_divider[nn_idx_divider == 0] = 1
        # nei_inds[nei_inds == -1] = 0

        gathered_xyz = index_points(sparse_xyz, nei_inds)
        # localized_xyz = gathered_xyz - dense_xyz.view(B, M, 1, D) #[B, M, K, D]
        localized_xyz = gathered_xyz - dense_xyz.unsqueeze(dim=2)
        gathered_norm = index_points(sparse_xyz_norm, nei_inds)

        if self.cfg.USE_PE:
            feat_pe = self.pe_convs(localized_xyz)
#        feat_pe = localized_xyz  # [B, in_ch+D, K, M]
#        for i, conv in enumerate(self.pe_convs):
#            feat_pe = F.relu(conv(feat_pe), inplace=True)
        if self.cfg.USE_VI is True:
          if vi_features is None:
              weightNetInput = VI_coordinate_transform(localized_xyz, gathered_norm, dense_xyz_norm, K)
          else:
              weightNetInput = vi_features
        else:
            weightNetInput = localized_xyz

        feats_x = self.unary1(sparse_feats)
#        feats_x = sparse_feats
        
        if not self.cfg.USE_CUDA_KERNEL:
            gathered_feat = index_points(feats_x, nei_inds)  # [B, M, K, in_ch]
            gathered_feat = torch.cat([gathered_feat, feat_pe], dim=-1)

        weights = self.weightnet(weightNetInput)

        if self.cfg.TIME:
            torch.cuda.synchronize()
            t5 = time()
            print("weightnet time: ", t5 - t4)
        if self.cfg.USE_CUDA_KERNEL:
            feats_x = feats_x.contiguous()
            nei_inds = nei_inds.contiguous()
            weights = weights.contiguous()
            feat_pe = feat_pe.contiguous()
            new_feat = PConv.forward(feats_x, nei_inds, weights, feat_pe)
        else:
            new_feat = torch.matmul(input=gathered_feat.permute(0, 1,3, 2), other=weights).view(B, M, -1)

        new_feat = F.relu(self.linear(new_feat), inplace=True)
        
        if dense_feats is not None:
            # new_feat = torch.cat([new_feat, dense_feats.permute(0, 2, 1)], dim = 1)
            new_feat = new_feat + dense_feats

        # Dropout
        new_feat = self.dropout(new_feat)

        for i, conv in enumerate(self.mlp2_convs):
            new_feat = F.relu(conv(new_feat), inplace=True)

        return new_feat, weightNetInput


# The backbone-only part of PCF
class PCF_Backbone(nn.Module):
    def __init__(self, cfg, input_feat_dim = 3):
        super(PCF_Backbone, self).__init__()

        self.cfg = cfg
        self.total_level = cfg.num_level
        self.guided_level = cfg.guided_level
        
        self.input_feat_dim = input_feat_dim + 3 if cfg.USE_XYZ else input_feat_dim


        self.relu = torch.nn.ReLU(inplace=True)

        if cfg.USE_VI is True:
            weightnet_input_dim = cfg.point_dim + 9
        else:
            weightnet_input_dim = cfg.point_dim
        weightnet = [weightnet_input_dim, cfg.mid_dim[0]] # 2 hidden layer
        weightnet_start = [weightnet_input_dim, cfg.mid_dim[0]]

        if cfg.use_level_1:
            self.selfpointconv = PointConv(self.input_feat_dim, cfg.base_dim, [], cfg, weightnet_start)
            self.selfpointconv_res1 = PointConvStridePE(cfg.base_dim, cfg.base_dim, cfg, weightnet_start)
            self.selfpointconv_res2 = PointConvStridePE(cfg.base_dim, cfg.base_dim, cfg, weightnet_start)
        else:
            self.selfmlp = Linear_BN(self.input_feat_dim, cfg.base_dim, bn_ver = '1d')
#            cfg.feat_dim[0] = self.input_feat_dim


        self.pointconv = nn.ModuleList()
        self.pointconv_res = nn.ModuleList()

        for i in range(1, self.total_level):
            in_ch = cfg.feat_dim[i - 1]
            out_ch = cfg.feat_dim[i]
            weightnet = [weightnet_input_dim, cfg.mid_dim[i]]

            if i <= self.guided_level:
                self.pointconv.append(PointConvStridePE(in_ch, out_ch, cfg, weightnet))
            else:
              if self.cfg.transformer_type == 'PCF':
                self.pointconv.append(GuidedConvStridePENewAfter(in_ch, out_ch, cfg, weightnet, cfg.num_heads))
              else:
                self.pointconv.append(PointTransformerLayer(in_ch, out_ch, cfg.num_heads))

            if self.cfg.resblocks[i] == 0:
                self.pointconv_res.append(nn.ModuleList([]))
            else:
                res_blocks = nn.ModuleList()
                for _ in range(self.cfg.resblocks[i]):
                    if i <= self.guided_level:
                        res_blocks.append(PointConvStridePE(out_ch, out_ch, cfg, weightnet))
                    else:
                      if self.cfg.transformer_type == 'PCF':
                        res_blocks.append(GuidedConvStridePENewAfter(out_ch, out_ch, cfg, weightnet, cfg.num_heads))
                      else:
                        res_blocks.append(PointTransformerLayer(out_ch, out_ch, cfg.num_heads))
                self.pointconv_res.append(res_blocks)
#    @profile
    def forward(self, features, pointclouds, edges_self, edges_forward, norms):
        # import ipdb; ipdb.set_trace()
        if self.cfg.TIME:
            print("number_of_points: ", features.shape)

            torch.cuda.synchronize()
            t0 = time()
        #encode pointwise info
        pointwise_feat = torch.cat([features, pointclouds[0]], -1) if self.cfg.USE_XYZ else features
        if self.cfg.TIME:
            torch.cuda.synchronize()
            t1 = time()
            print("pointwise time: ", t1 - t0)

        # level 1 conv, this helps performance significantly on 5cm/10cm inputs but have relatively small use on 2cm
        if self.cfg.use_level_1:
            pointwise_feat, vi_features = self.selfpointconv(pointclouds[0], pointwise_feat, edges_self[0], norms[0])
            pointwise_feat, _ = self.selfpointconv_res1(pointclouds[0], pointwise_feat, edges_self[0], norms[0], vi_features = vi_features)
            pointwise_feat, _ = self.selfpointconv_res2(pointclouds[0], pointwise_feat, edges_self[0], norms[0], vi_features = vi_features)   
        else:
        # if don't use level 1 convs, then just simply do a linear layer to increase the feature dimensionality
            pointwise_feat = F.relu(self.selfmlp(pointwise_feat),inplace=True)
        if self.cfg.TIME:
            torch.cuda.synchronize()
            t2 = time()

            print("level1 selfpointconv time: ", t2 - t1)

    
        feat_list = [pointwise_feat]
        for i, pointconv in enumerate(self.pointconv):
          if self.cfg.TIME:
             torch.cuda.synchronize()
             old_t = t2
             t2 = time()
             print("pointconv in resblock ", i," time: ", t2 - old_t)
          if self.cfg.transformer_type != 'PCF':
              sparse_feat = pointconv(pointclouds[i], feat_list[-1], edges_forward[i], pointclouds[i+1])
          else:
              sparse_feat, _ = pointconv(pointclouds[i],feat_list[-1], edges_forward[i], norms[i], pointclouds[i+1], norms[i+1])
          #print(sparse_feat.shape)
          # There is the need to recompute VI features from the neighbors at this level rather than from the previous level, hence need
          # to recompute VI features in the first residual block
          vi_features = None
          for res_block in self.pointconv_res[i]:
              if self.cfg.transformer_type != 'PCF':
                  sparse_feat = res_block(pointclouds[i+1],sparse_feat, edges_self[i+1])
              else:
                if vi_features is not None:
                  sparse_feat, _ = res_block(pointclouds[i+1], sparse_feat, edges_self[i+1], norms[i+1], vi_features = vi_features)
                else:
                  sparse_feat, vi_features = res_block(pointclouds[i+1], sparse_feat, edges_self[i+1], norms[i+1])


          feat_list.append(sparse_feat)
           # print(sparse_feat.shape)
          if self.cfg.TIME:
             torch.cuda.synchronize()
             old_t = t2
             t2 = time()
             print("resblock ", i," time: ", t2 - old_t)
        

        return feat_list

def PCF_Tiny(input_grid_size, base_dim = 64):
    cfg = EasyDict()
    cfg = get_default_configs(cfg, num_level = 5, base_dim = base_dim)
    cfg.guided_level = 0
    cfg.num_heads = 1
    cfg.resblocks = [0,1,1,1,1]
    cfg.mid_dim = [4,4,4,4,4]
    cfg.grid_size = [input_grid_size, input_grid_size*2,input_grid_size*4,input_grid_size*8,input_grid_size*16]
    return PCF_Backbone(cfg), cfg

def PCF_Small(input_grid_size, base_dim = 64):
    cfg = EasyDict()
    cfg = get_default_configs(cfg, num_level = 5, base_dim = base_dim)
    cfg.guided_level = 0
    cfg.num_heads = 8
    cfg.resblocks = [0,2,2,2,2]
    cfg.mid_dim = [4,4,4,4,4]
    cfg.grid_size = [input_grid_size, input_grid_size*2,input_grid_size*4,input_grid_size*8,input_grid_size*16]
    return PCF_Backbone(cfg), cfg

def PCF_Normal(input_grid_size, base_dim = 64):
    cfg = EasyDict()
    cfg = get_default_configs(cfg, num_level = 5, base_dim = base_dim)
    cfg.guided_level = 0
    cfg.num_heads = 8
    cfg.resblocks = [0, 2,4,6,6]
    cfg.grid_size = [input_grid_size, input_grid_size*2,input_grid_size*4,input_grid_size*8,input_grid_size*16]
    cfg.mid_dim = [16,16,16,16,16]
    return PCF_Backbone(cfg), cfg

def PCF_Large(input_grid_size, base_dim = 64):
    cfg = EasyDict()
    cfg = get_default_configs(cfg, num_level = 6, base_dim = base_dim)
    cfg.guided_level = 0
    cfg.num_heads = 8
    cfg.resblocks = [0,2,4,6,6, 2]
    cfg.grid_size = [input_grid_size, input_grid_size*2.5,input_grid_size*5,input_grid_size*10,input_grid_size*20, input_grid_size*40]
    cfg.mid_dim = [16,16,16,16,16,16]
    return PCF_Backbone(cfg), cfg


# The entire model, including the backbone and the segmentation decoder
class PointConvFormer_Segmentation(nn.Module):
    def __init__(self, cfg):
        super(PointConvFormer_Segmentation, self).__init__()

        self.cfg = cfg
        self.total_level = cfg.num_level
     
        self.pcf_backbone = PCF_Backbone(cfg)
#        self.pcf_backbone,pcf_cfg = PCF_Normal(cfg.grid_size[0], cfg.base_dim)
        if cfg.USE_VI is True:
            weightnet = [cfg.point_dim+9, cfg.mid_dim_back] # 2 hidden layer
        else:
            weightnet = [cfg.point_dim, cfg.mid_dim_back]

        self.pointdeconv = nn.ModuleList()
        self.pointdeconv_res = nn.ModuleList()

        for i in range(self.total_level - 2, -1, -1):
            in_ch = cfg.feat_dim[i + 1]
            if i==0:
                out_ch = cfg.base_dim
            else:
                out_ch = cfg.feat_dim[i]

            mlp2 = [out_ch, out_ch]
#            self.pointdeconv.append(GuidedConvStridePENewAfter(in_ch, out_ch, cfg, weightnet, cfg.num_heads))
#            if i==0:
#                cfg2 = EasyDict(d = cfg.copy())
#                cfg2.USE_PE = False
#                self.pointdeconv.append(PointConvTransposePE(in_ch, out_ch, cfg2, weightnet, mlp2))
#            else:
            self.pointdeconv.append(PointConvTransposePE(in_ch, out_ch, cfg, weightnet, mlp2))

            if self.cfg.resblocks[i] == 0:
                self.pointdeconv_res.append(nn.ModuleList([]))
            else:
                res_blocks = nn.ModuleList()
                for _ in range(self.cfg.resblocks_back[i]):
                    res_blocks.append(PointConvStridePE(out_ch, out_ch, cfg, weightnet))
                self.pointdeconv_res.append(res_blocks)

        #pointwise_decode
        self.fc1 = Linear_BN(cfg.base_dim, cfg.base_dim,bn_ver='1d')
        self.dropout_fc = torch.nn.Dropout(p=cfg.dropout_fc) if cfg.dropout_fc > 0. else nn.Identity()
        self.fc2 = nn.Linear(cfg.base_dim, cfg.num_classes)
#    @profile
    def forward(self, features, pointclouds, edges_self, edges_forward, edges_propagate, norms):
#      with LineProfiler()  as prof:
        # import ipdb; ipdb.set_trace()
        if self.cfg.TIME:
            print("number_of_points: ", features.shape)
            torch.cuda.synchronize()
            t2 = time()
#        with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as prof:
        feat_list = self.pcf_backbone(features, pointclouds, edges_self, edges_forward, norms)
#        print(prof.key_averages(group_by_stack_n=7).table(sort_by="cuda_time_total",row_limit=50))
#        prof.export_stacks("profiler_stacks.txt", "self_cuda_time_total")

        sparse_feat = feat_list[-1]
        for i, pointdeconv in enumerate(self.pointdeconv):
            cur_level = self.total_level - 2 - i

#            sparse_feat, _ = pointdeconv(pointclouds[cur_level+1], sparse_feat, edges_propagate[cur_level], norms[cur_level+1], pointclouds[cur_level], norms[cur_level])
#            sparse_feat = sparse_feat + feat_list[cur_level]
            sparse_feat, _ = pointdeconv(pointclouds[cur_level+1], sparse_feat, edges_propagate[cur_level], norms[cur_level+1], pointclouds[cur_level], feat_list[cur_level], norms[cur_level])
#            else:
#                sparse_feat, _ = pointdeconv(pointclouds[cur_level+1], sparse_feat, edges_propagate[cur_level], norms[cur_level+1], pointclouds[cur_level], None, norms[cur_level])

            vi_features = None
            for res_block in self.pointdeconv_res[i]:
                nei_inds = edges_self[cur_level]
                if vi_features is not None:
                    sparse_feat, _ = res_block(pointclouds[cur_level], sparse_feat, edges_self[cur_level], norms[cur_level], vi_features = vi_features)
                else:
                    sparse_feat, vi_features = res_block(pointclouds[cur_level], sparse_feat, edges_self[cur_level], norms[cur_level])

            feat_list[cur_level] = sparse_feat
            # print(sparse_feat.shape)

            if self.cfg.TIME:
                torch.cuda.synchronize()
                old_t = t2
                t2 = time()
                print("back resblock ", i," time: ", t2 - old_t)
        

        fc = self.dropout_fc(F.relu(self.fc1(sparse_feat)))
        fc = self.fc2(fc)
#      prof.display()
        if self.cfg.TIME:
            torch.cuda.synchronize()
            old_t = t2
            t2 = time()
            print("last linear time: ", t2 - old_t)
    
        return fc
