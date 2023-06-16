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

def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S] / [B, S, K]
    Return:
        new_points:, indexed points data, [B, S, C] / [B, S, K, C]
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

# Batch Normalization: if use_bn is true, then this is just batch normalization
# if use_bn is false, then it's a simple bias term
class BatchNormBlock(nn.Module):

    def __init__(self, in_dim, use_bn, bn_momentum):
        """
        Initialize a batch normalization block. If network does not use batch normalization, replace with biases.
        :param in_dim: dimension input features
        :param use_bn: boolean indicating if we use Batch Norm
        :param bn_momentum: Batch norm momentum
        """
        super(BatchNormBlock, self).__init__()
        self.bn_momentum = bn_momentum
        self.use_bn = use_bn
        self.in_dim = in_dim
        if self.use_bn:
            self.batch_norm = nn.BatchNorm1d(in_dim, momentum=bn_momentum)
            #self.batch_norm = nn.InstanceNorm1d(in_dim, momentum=bn_momentum)
        else:
            self.bias = Parameter(torch.zeros(in_dim, dtype=torch.float32), requires_grad=True)
        return

    def reset_parameters(self):
        nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.use_bn:

            # x = x.unsqueeze(2)
            x = x.transpose(1, 2)
            x = self.batch_norm(x)
            x = x.transpose(1, 2)
            return x
        else:
            return x + self.bias

    def __repr__(self):
        return 'BatchNormBlock(in_feat: {:d}, momentum: {:.3f}, only_bias: {:s})'.format(self.in_dim,
                                                                                         self.bn_momentum,
                                                                                         str(not self.use_bn))

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
        self.mlp = nn.Linear(in_dim, out_dim, bias=False)
        self.batch_norm = BatchNormBlock(out_dim, self.use_bn, self.bn_momentum)
        if not no_relu:
            self.leaky_relu = nn.LeakyReLU(0.1)
        return

    def forward(self, x):
        x = self.mlp(x)
        x = self.batch_norm(x)
        if not self.no_relu:
            x = self.leaky_relu(x)
        return x

    def __repr__(self):
        return 'UnaryBlock(in_feat: {:d}, out_feat: {:d}, BN: {:s}, ReLU: {:s})'.format(self.in_dim,
                                                                                        self.out_dim,
                                                                                        str(self.use_bn),
                                                                                        str(not self.no_relu))


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
        mlp_dim = [self.dim, 8, 8, num_heads]
        for ch_in, ch_out in zip(mlp_dim[:-1], mlp_dim[1:]):
            self.mlp.append(nn.Linear(ch_in, ch_out))
            if cfg.BATCH_NORM:
                self.mlp_bns.append(nn.BatchNorm1d(ch_out))

        # self.pos_encoding = nn.Linear(self.dim, num_heads)

    def forward(self, guidance_query, guidance_key, pos_encoding=None):

        # import ipdb; ipdb.set_trace()
        # attention bxcxkxn
        batch_dim, _, k, n = guidance_query.shape

        guidance_query = guidance_query.view(batch_dim, self.dim, k, n).permute(0, 2, 3, 1)
        guidance_key = guidance_key.view(batch_dim, self.dim, k, n).permute(0, 2, 3, 1)

        # scores = torch.einsum('bdhkn, bdhkn->bhkn', guidance_query, guidance_key) / self.dim**.5
        # scores = scores.unsqueeze(-1)
        scores = self.layer_norm_q(guidance_query) - self.layer_norm_k(guidance_key)
        scores = scores if pos_encoding is None else scores + pos_encoding.permute(0, 2, 3, 1)

        for i, conv in enumerate(self.mlp):
            if self.cfg.BATCH_NORM:
                bn = self.mlp_bns[i]
                # print(bn)
                scores = conv(scores).view(batch_dim * k, n, -1).permute(0, 2, 1)
                # print(scores.shape)
                scores = bn(scores).permute(0, 2, 1).view(batch_dim, k, n, -1)
            else:
                scores = conv(scores)

            if i == len(self.mlp) - 1:
                # pos_encoding = self.pos_encoding(pos_encoding.permute(0, 2, 3, 1))
                # scores = torch.sigmoid(scores + pos_encoding)
                scores = torch.sigmoid(scores)
                # scores = torch.nn.functional.softmax(scores, dim = 1)
                # scores = F.relu(scores)
                # scores = torch.tanh(scores).squeeze(-1)
            else:
                scores = F.relu(scores, inplace=True)

        #import ipdb; ipdb.set_trace()

        scores = scores.permute(0, 3, 1, 2) # b, nh, k, n

        return scores


class MultiHeadGuidanceNewV1VI(nn.Module):
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
        mlp_dim = [self.dim, 8, 8, num_heads]
        for ch_in, ch_out in zip(mlp_dim[:-1], mlp_dim[1:]):
            self.mlp.append(nn.Linear(ch_in, ch_out))
            if cfg.BATCH_NORM:
                self.mlp_bns.append(nn.BatchNorm1d(ch_out))

        # self.pos_encoding = nn.Linear(self.dim, num_heads)

    def forward(self, guidance_query, guidance_key, pos_encoding=None):

        # import ipdb; ipdb.set_trace()
        # attention bxcxkxn
        batch_dim, dim, k, n = guidance_query.shape

        guidance_query = guidance_query.view(batch_dim, dim, k, n).permute(0, 2, 3, 1)
        guidance_key = guidance_key.view(batch_dim, dim, k, n).permute(0, 2, 3, 1)

        # scores = torch.einsum('bdhkn, bdhkn->bhkn', guidance_query, guidance_key) / self.dim**.5
        # scores = scores.unsqueeze(-1)
        scores = self.layer_norm_q(guidance_query) - self.layer_norm_k(guidance_key)
        scores = scores if pos_encoding is None else torch.cat((scores, pos_encoding.permute(0, 2, 3, 1)), dim = -1)

        for i, conv in enumerate(self.mlp):
            if self.cfg.BATCH_NORM:
                bn = self.mlp_bns[i]
                # print(bn)
                scores = conv(scores).view(batch_dim * k, n, -1).permute(0, 2, 1)
                # print(scores.shape)
                scores = bn(scores).permute(0, 2, 1).view(batch_dim, k, n, -1)
            else:
                scores = conv(scores)

            if i == len(self.mlp) - 1:
                # pos_encoding = self.pos_encoding(pos_encoding.permute(0, 2, 3, 1))
                # scores = torch.sigmoid(scores + pos_encoding)
                scores = torch.sigmoid(scores)
                # scores = torch.nn.functional.softmax(scores, dim = 1)
                # scores = F.relu(scores)
                # scores = torch.tanh(scores).squeeze(-1)
            else:
                scores = F.relu(scores, inplace=True)

        #import ipdb; ipdb.set_trace()

        scores = scores.permute(0, 3, 1, 2) # b, nh, k, n

        return scores


class MultiHeadGuidanceQK(nn.Module):
    """ Multi-head guidance to increase model expressivitiy"""
    def __init__(self, cfg, num_heads: int, num_hiddens: int, key_dim: int):
        super().__init__()
        assert num_hiddens % num_heads == 0, 'num_hiddens: %d, num_heads: %d'%(num_hiddens, num_heads)
        self.cfg = cfg
        self.dim = num_hiddens
        self.num_heads = num_heads
        # self.key_dim = key_dim 
        self.key_dim = key_dim = num_hiddens // num_heads
        self.scale = self.key_dim ** -0.5
        self.dh = int(key_dim * num_heads)
        self.nh_kd = nh_kd = key_dim * num_heads 
        h = self.dh + nh_kd
        self.qk_linear = nn.Linear(self.dim, h)
        self.qk_bn = nn.BatchNorm1d(h)
#        self.v_linear = nn.Linear(key_dim, int(self.dim / num_heads))
        self.pe_linear = nn.Linear(self.dim, key_dim)
        

    def forward(self, x, nei_inds, pe = None):
        # x: b, n, c

        # import ipdb; ipdb.set_trace()
        # compute q, k
        B, N, C = x.shape 
        nei_size = nei_inds.shape[-1]
        qk = self.qk_linear(x)
        qk = self.qk_bn(qk.flatten(0, 1)).reshape_as(qk)
        q, k = qkv.view(B, N, self.num_heads, -1).split([self.key_dim, self.key_dim], dim = 3)

        # gather point with knn neighbourhood
        gathered_k = index_points(k.flatten(2, 3), nei_inds).view(B, N, nei_size, self.num_heads, -1) # b, n, k, nh, c
        gathered_q = q.unsqueeze(2).repeat(1, 1, nei_size, 1, 1)

        # compute attention 
        attn_score = (gathered_k * gathered_q).sum(-1, keepdim=True) * self.scale 
        if pe is not None:
            pe = self.pe_linear(pe)
            attn_score += pe
        # attn_score = attn_score.sigmoid() # b, n, nei_size, nh, 1
        # attn_score = F.softmax(attn_score, dim = 2)
        attn_score = torch.sigmoid(attn_score)

        # prepare v 
#        gathered_v = index_points(v.flatten(2, 3), nei_inds).view(B, N, nei_size, self.num_heads, -1)
#        attned_v = self.v_linear(gathered_v * attn_score).view(B, N, nei_size, self.dim)

        return attn_score



class BatchNormBlock(nn.Module):

    def __init__(self, in_dim, use_bn, bn_momentum):
        """
        Initialize a batch normalization block. If network does not use batch normalization, replace with biases.
        :param in_dim: dimension input features
        :param use_bn: boolean indicating if we use Batch Norm
        :param bn_momentum: Batch norm momentum
        """
        super(BatchNormBlock, self).__init__()
        self.bn_momentum = bn_momentum
        self.use_bn = use_bn
        self.in_dim = in_dim
        if self.use_bn:
            self.batch_norm = nn.BatchNorm1d(in_dim, momentum=bn_momentum)
            #self.batch_norm = nn.InstanceNorm1d(in_dim, momentum=bn_momentum)
        else:
            self.bias = Parameter(torch.zeros(in_dim, dtype=torch.float32), requires_grad=True)
        return

    def reset_parameters(self):
        nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.use_bn:

            # x = x.unsqueeze(2)
            x = x.transpose(1, 2)
            x = self.batch_norm(x)
            x = x.transpose(1, 2)
            return x
        else:
            return x + self.bias

    def __repr__(self):
        return 'BatchNormBlock(in_feat: {:d}, momentum: {:.3f}, only_bias: {:s})'.format(self.in_dim,
                                                                                         self.bn_momentum,
                                                                                         str(not self.use_bn))


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
        self.mlp = nn.Linear(in_dim, out_dim, bias=False)
        self.batch_norm = BatchNormBlock(out_dim, self.use_bn, self.bn_momentum)
        if not no_relu:
            self.leaky_relu = nn.LeakyReLU(0.1)
        return

    def forward(self, x):
        x = self.mlp(x)
        x = self.batch_norm(x)
        if not self.no_relu:
            x = self.leaky_relu(x)
        return x

    def __repr__(self):
        return 'UnaryBlock(in_feat: {:d}, out_feat: {:d}, BN: {:s}, ReLU: {:s})'.format(self.in_dim,
                                                                                        self.out_dim,
                                                                                        str(self.use_bn),
                                                                                        str(not self.no_relu))


class WeightNet(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_unit=[8, 8]):
        super(WeightNet, self).__init__()

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        if hidden_unit is None or len(hidden_unit) == 0:
            # self.mlp_convs.append(nn.Conv2d(in_channel, out_channel, 1))
            self.mlp_convs.append(nn.Linear(in_channel, out_channel))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        else:
            # self.mlp_convs.append(nn.Conv2d(in_channel, hidden_unit[0], 1))
            self.mlp_convs.append(nn.Linear(in_channel, hidden_unit[0]))
            self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[0]))
            for i in range(1, len(hidden_unit)):
                # self.mlp_convs.append(nn.Conv2d(hidden_unit[i - 1], hidden_unit[i], 1))
                self.mlp_convs.append(nn.Linear(hidden_unit[i - 1], hidden_unit[i]))
                self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[i]))
            # self.mlp_convs.append(nn.Conv2d(hidden_unit[-1], out_channel, 1))
            self.mlp_convs.append(nn.Linear(hidden_unit[-1], out_channel))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))

    def forward(self, localized_xyz):
        # xyz : BxCxKxN

        weights = localized_xyz
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            weights = conv(weights.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
            weights = F.relu(bn(weights), inplace=True)
            # weights = F.relu(weights, inplace=True)

        return weights


class GuidedConvStridePENewAfter(nn.Module):
    def __init__(self, in_channel, out_channel, cfg, weightnet=[9, 16], num_heads=4):
        super(GuidedConvStridePENewAfter, self).__init__()
        self.cfg = cfg
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.num_heads = num_heads

        self.drop_path = DropPath(cfg.drop_path_rate) if cfg.drop_path_rate > 0. else nn.Identity()

        # positonal encoder
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_ch = 3
        for out_ch in [out_channel//4, out_channel//4]:
            self.mlp_convs.append(nn.Conv2d(last_ch, out_ch, 1))
            if cfg.BATCH_NORM:
                self.mlp_bns.append(nn.BatchNorm2d(out_ch))
            last_ch = out_ch

        # First downscaling mlp
        if in_channel != out_channel // 4:
            self.unary1 = UnaryBlock(in_channel, out_channel // 4, use_bn=True, bn_momentum=0.1)
        else:
            self.unary1 = nn.Identity()

        # check last_ch % num_heads == 0
        assert (out_channel // 2) % num_heads == 0
        self.guidance_weight = MultiHeadGuidanceNewV1(cfg, num_heads, out_channel // 2)
        self.mix_linear = nn.Conv2d(out_channel // 2, out_channel // 2, 1)

        self.weightnet = WeightNet(weightnet[0], weightnet[1])
        self.linear = nn.Linear(out_channel // 2 * weightnet[-1], out_channel // 2)
        if cfg.BATCH_NORM:
            self.bn_linear = nn.BatchNorm1d(out_channel // 2)

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

    def forward(self, dense_xyz, sparse_xyz, dense_feats, nei_inds, dense_xyz_norm, sparse_xyz_norm):
        """
        dense_xyz: tensor (batch_size, num_points, 3)
        sparse_xyz: tensor (batch_size, num_points2, 3)
        dense_feats: tensor (batch_size, num_points, num_dims)
        nei_inds: tensor (batch_size, num_points2, K)
        """
        B, N, D = dense_xyz.shape
        _, M, _ = sparse_xyz.shape
        _, _, in_ch = dense_feats.shape
        _, _, K = nei_inds.shape

        # first downscaling mlp
        feats_x = self.unary1(dense_feats)

        gathered_xyz = index_points(dense_xyz, nei_inds)
        # localized_xyz = gathered_xyz - sparse_xyz.view(B, M, 1, D) #[B, M, K, D]
        localized_xyz = gathered_xyz - sparse_xyz.unsqueeze(dim=2)
        gathered_norm = index_points(dense_xyz_norm, nei_inds)

        feat_pe = localized_xyz.permute(0, 3, 2, 1)  # [B, in_ch+D, K, M]
        for i, conv in enumerate(self.mlp_convs):
            if self.cfg.BATCH_NORM:
                bn = self.mlp_bns[i]
                feat_pe = F.relu(bn(conv(feat_pe)), inplace=True)
            else:
                feat_pe = F.relu(conv(feat_pe), inplace=True)


        n_alpha = gathered_norm
        n_miu = sparse_xyz_norm
        r_miu = localized_xyz
        r_hat = F.normalize(r_miu, dim=3)
        v_miu = n_miu.unsqueeze(dim=2) - torch.matmul(n_miu.unsqueeze(dim=2), r_hat.permute(0, 1, 3, 2)).permute(0, 1, 3, 2) * r_hat
        v_miu = F.normalize(v_miu, dim=3)
        w_miu = torch.cross(r_hat, v_miu, dim=3)
        w_miu = F.normalize(w_miu, dim=3)
        theta1 = torch.matmul(n_alpha, n_miu.unsqueeze(dim=3))
        theta2 = torch.matmul(r_hat, n_miu.unsqueeze(dim=3))
        theta3 = torch.sum(r_hat * n_alpha, dim=3, keepdim=True)
        theta4 = torch.matmul(r_miu, n_miu.unsqueeze(dim=3))
        theta5 = torch.sum(n_alpha * r_hat, dim=3, keepdim=True)
        theta6 = torch.sum(n_alpha * v_miu, dim=3, keepdim=True)
        theta7 = torch.sum(n_alpha * w_miu, dim=3, keepdim=True)
        theta8 = torch.sum(r_miu * torch.cross(n_alpha, n_miu.unsqueeze(dim=2).repeat(1, 1, K, 1), dim=3), dim=3, keepdim=True)
        theta9 = torch.norm(r_miu, dim=3, keepdim=True)
        weightNetInput = torch.cat(
            [theta1, theta2, theta3, theta4, theta5, theta6, theta7, theta8, theta9, localized_xyz], dim=3).contiguous()

        gathered_feat = index_points(feats_x, nei_inds)  # [B, M, K, in_ch]
        # new_feat = gathered_feat.permute(0, 3, 2, 1)
        new_feat = torch.cat([gathered_feat, feat_pe.permute(0, 3, 2, 1).contiguous()], dim=-1).permute(0, 3, 2, 1)
        
        # import ipdb; ipdb.set_trace()
        guidance_feature = new_feat
        guidance_query = guidance_feature # b c k m
        guidance_key = guidance_feature.max(dim=2, keepdim=True)[0].repeat(1, 1, K, 1)
        guidance_score = self.guidance_weight(guidance_query, guidance_key).unsqueeze(1) # b num_heads k n

        # apply guide to features
        new_feat = self.mix_linear(new_feat)
        new_feat = (new_feat.view(B, -1, self.num_heads, K, M) * guidance_score).view(B, -1, K, M).contiguous()

        weightNetInput = weightNetInput.permute(0, 3, 2, 1)
        weights = self.weightnet(weightNetInput).contiguous()

        # localized_xyz = localized_xyz.permute(0, 3, 2, 1)
        # weights = self.weightnet(localized_xyz)*nei_inds_mask.permute(0,2,1).unsqueeze(dim=1)

        new_feat = torch.matmul(input=new_feat.permute(0, 3, 1, 2).contiguous(), other=weights.permute(0, 3, 2, 1).contiguous()).view(B, M, -1)
        # new_feat = new_feat/nn_idx_divider.unsqueeze(dim=-1)
        new_feat = self.linear(new_feat)
        if self.cfg.BATCH_NORM:
            new_feat = self.bn_linear(new_feat.permute(0, 2, 1))
            new_feat = F.relu(new_feat, inplace=True).permute(0, 2, 1)
        else:
            new_feat = F.relu(new_feat, inplace=True)

        # Dropout
        new_feat = self.dropout(new_feat)

        # Second upscaling mlp
        new_feat = self.unary2(new_feat)

        sparse_feats = torch.max(index_points(dense_feats, nei_inds), dim = 2)[0]

        shortcut = self.unary_shortcut(sparse_feats)

        new_feat = self.leaky_relu(self.drop_path(new_feat) + shortcut)

        return new_feat


class GuidedConvStridePENewAfterVI(nn.Module):
    def __init__(self, in_channel, out_channel, cfg, weightnet=[9, 16], num_heads=4):
        super(GuidedConvStridePENewAfterVI, self).__init__()
        self.cfg = cfg
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.num_heads = num_heads

        self.drop_path = DropPath(cfg.drop_path_rate) if cfg.drop_path_rate > 0. else nn.Identity()

        # positonal encoder
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_ch = weightnet[0]
        for out_ch in [out_channel//4, out_channel//4]:
            self.mlp_convs.append(nn.Conv2d(last_ch, out_ch, 1))
            if cfg.BATCH_NORM:
                self.mlp_bns.append(nn.BatchNorm2d(out_ch))
            last_ch = out_ch

        # First downscaling mlp
        if in_channel != out_channel // 4:
            self.unary1 = UnaryBlock(in_channel, out_channel // 4, use_bn=True, bn_momentum=0.1)
        else:
            self.unary1 = nn.Identity()

        # check last_ch % num_heads == 0
        assert (out_channel // 2) % num_heads == 0
        self.guidance_weight = MultiHeadGuidanceNewV1VI(cfg, num_heads, out_channel // 2)
        self.mix_linear = nn.Conv2d(out_channel // 2, out_channel // 2, 1)

        self.weightnet = WeightNet(weightnet[0], weightnet[1])
        self.linear = nn.Linear(out_channel // 2 * weightnet[-1], out_channel // 2)
        if cfg.BATCH_NORM:
            self.bn_linear = nn.BatchNorm1d(out_channel // 2)

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

    def forward(self, dense_xyz, sparse_xyz, dense_feats, nei_inds, dense_xyz_norm, sparse_xyz_norm):
        """
        dense_xyz: tensor (batch_size, num_points, 3)
        sparse_xyz: tensor (batch_size, num_points2, 3)
        dense_feats: tensor (batch_size, num_points, num_dims)
        nei_inds: tensor (batch_size, num_points2, K)
        """
        B, N, D = dense_xyz.shape
        _, M, _ = sparse_xyz.shape
        _, _, in_ch = dense_feats.shape
        _, _, K = nei_inds.shape

        # first downscaling mlp
        feats_x = self.unary1(dense_feats)

        gathered_xyz = index_points(dense_xyz, nei_inds)
        # localized_xyz = gathered_xyz - sparse_xyz.view(B, M, 1, D) #[B, M, K, D]
        localized_xyz = gathered_xyz - sparse_xyz.unsqueeze(dim=2)
        gathered_norm = index_points(dense_xyz_norm, nei_inds)

        n_alpha = gathered_norm
        n_miu = sparse_xyz_norm
        r_miu = localized_xyz
        r_hat = F.normalize(r_miu, dim=3)
        v_miu = n_miu.unsqueeze(dim=2) - torch.matmul(n_miu.unsqueeze(dim=2), r_hat.permute(0, 1, 3, 2)).permute(0, 1, 3, 2) * r_hat
        v_miu = F.normalize(v_miu, dim=3)
        w_miu = torch.cross(r_hat, v_miu, dim=3)
        w_miu = F.normalize(w_miu, dim=3)
        theta1 = torch.matmul(n_alpha, n_miu.unsqueeze(dim=3))
        theta2 = torch.matmul(r_hat, n_miu.unsqueeze(dim=3))
        theta3 = torch.sum(r_hat * n_alpha, dim=3, keepdim=True)
        theta4 = torch.matmul(r_miu, n_miu.unsqueeze(dim=3))
        theta5 = torch.sum(n_alpha * r_hat, dim=3, keepdim=True)
        theta6 = torch.sum(n_alpha * v_miu, dim=3, keepdim=True)
        theta7 = torch.sum(n_alpha * w_miu, dim=3, keepdim=True)
        theta8 = torch.sum(r_miu * torch.cross(n_alpha, n_miu.unsqueeze(dim=2).repeat(1, 1, K, 1), dim=3), dim=3, keepdim=True)
        theta9 = torch.norm(r_miu, dim=3, keepdim=True)
        weightNetInput = torch.cat(
            [theta1, theta2, theta3, theta4, theta5, theta6, theta7, theta8, theta9, localized_xyz], dim=3).contiguous()

        feat_pe = weightNetInput.permute(0, 3, 2, 1)  # [B, in_ch+D, K, M]
        for i, conv in enumerate(self.mlp_convs):
            if self.cfg.BATCH_NORM:
                bn = self.mlp_bns[i]
                feat_pe = F.relu(bn(conv(feat_pe)), inplace=True)
            else:
                feat_pe = F.relu(conv(feat_pe), inplace=True)

        gathered_feat = index_points(feats_x, nei_inds)  # [B, M, K, in_ch]
        # new_feat = gathered_feat.permute(0, 3, 2, 1)
        new_feat = torch.cat([gathered_feat, feat_pe.permute(0, 3, 2, 1).contiguous()], dim=-1).permute(0, 3, 2, 1)
        
        # import ipdb; ipdb.set_trace()
        guidance_feature = gathered_feat.permute(0, 3, 2, 1)
        guidance_query = guidance_feature # b c k m
        guidance_key = guidance_feature.max(dim=2, keepdim=True)[0].repeat(1, 1, K, 1)
        guidance_score = self.guidance_weight(guidance_query, guidance_key, pos_encoding=feat_pe).unsqueeze(1) # b num_heads k n

        # apply guide to features
        new_feat = self.mix_linear(new_feat)
        new_feat = (new_feat.view(B, -1, self.num_heads, K, M) * guidance_score).view(B, -1, K, M).contiguous()

        weightNetInput = weightNetInput.permute(0, 3, 2, 1)
        weights = self.weightnet(weightNetInput).contiguous()

        # localized_xyz = localized_xyz.permute(0, 3, 2, 1)
        # weights = self.weightnet(localized_xyz)*nei_inds_mask.permute(0,2,1).unsqueeze(dim=1)

        new_feat = torch.matmul(input=new_feat.permute(0, 3, 1, 2).contiguous(), other=weights.permute(0, 3, 2, 1).contiguous()).view(B, M, -1)
        # new_feat = new_feat/nn_idx_divider.unsqueeze(dim=-1)
        new_feat = self.linear(new_feat)
        if self.cfg.BATCH_NORM:
            new_feat = self.bn_linear(new_feat.permute(0, 2, 1))
            new_feat = F.relu(new_feat, inplace=True).permute(0, 2, 1)
        else:
            new_feat = F.relu(new_feat, inplace=True)

        # Dropout
        new_feat = self.dropout(new_feat)

        # Second upscaling mlp
        new_feat = self.unary2(new_feat)

        sparse_feats = torch.max(index_points(dense_feats, nei_inds), dim = 2)[0]

        shortcut = self.unary_shortcut(sparse_feats)

        new_feat = self.leaky_relu(self.drop_path(new_feat) + shortcut)

        return new_feat


class GuidedConvStridePE(nn.Module):
    def __init__(self, in_channel, out_channel, cfg, weightnet=[9, 16], num_heads=4):
        super(GuidedConvStridePE, self).__init__()
        self.cfg = cfg
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.num_heads = num_heads

        self.drop_path = DropPath(cfg.drop_path_rate) if cfg.drop_path_rate > 0. else nn.Identity()

        # positonal encoder
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_ch = 3
        for out_ch in [out_channel//4, out_channel//4]:
            self.mlp_convs.append(nn.Conv2d(last_ch, out_ch, 1))
            if cfg.BATCH_NORM:
                self.mlp_bns.append(nn.BatchNorm2d(out_ch))
            last_ch = out_ch

        # First downscaling mlp
        if in_channel != out_channel // 4:
            self.unary1 = UnaryBlock(in_channel, out_channel // 4, use_bn=True, bn_momentum=0.1)
        else:
            self.unary1 = nn.Identity()

        # check last_ch % num_heads == 0
        assert (out_channel // 2) % num_heads == 0
        self.guidance_weight = MultiHeadGuidanceNewV1(cfg, num_heads, out_channel // 2)

        self.weightnet = WeightNet(weightnet[0], weightnet[1])
        self.linear = nn.Linear(out_channel // 2 * weightnet[-1], out_channel // 2)
        if cfg.BATCH_NORM:
            self.bn_linear = nn.BatchNorm1d(out_channel // 2)

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

    def forward(self, dense_xyz, sparse_xyz, dense_feats, nei_inds, dense_xyz_norm, sparse_xyz_norm):
        """
        dense_xyz: tensor (batch_size, num_points, 3)
        sparse_xyz: tensor (batch_size, num_points2, 3)
        dense_feats: tensor (batch_size, num_points, num_dims)
        nei_inds: tensor (batch_size, num_points2, K)
        """
        B, N, D = dense_xyz.shape
        _, M, _ = sparse_xyz.shape
        _, _, in_ch = dense_feats.shape
        _, _, K = nei_inds.shape

        # first downscaling mlp
        feats_x = self.unary1(dense_feats)

        gathered_xyz = index_points(dense_xyz, nei_inds)
        # localized_xyz = gathered_xyz - sparse_xyz.view(B, M, 1, D) #[B, M, K, D]
        localized_xyz = gathered_xyz - sparse_xyz.unsqueeze(dim=2)
        gathered_norm = index_points(dense_xyz_norm, nei_inds)

        feat_pe = localized_xyz.permute(0, 3, 2, 1)  # [B, in_ch+D, K, M]
        for i, conv in enumerate(self.mlp_convs):
            if self.cfg.BATCH_NORM:
                bn = self.mlp_bns[i]
                feat_pe = F.relu(bn(conv(feat_pe)), inplace=True)
            else:
                feat_pe = F.relu(conv(feat_pe), inplace=True)


        n_alpha = gathered_norm
        n_miu = sparse_xyz_norm
        r_miu = localized_xyz
        r_hat = F.normalize(r_miu, dim=3)
        v_miu = n_miu.unsqueeze(dim=2) - torch.matmul(n_miu.unsqueeze(dim=2), r_hat.permute(0, 1, 3, 2)).permute(0, 1, 3, 2) * r_hat
        v_miu = F.normalize(v_miu, dim=3)
        w_miu = torch.cross(r_hat, v_miu, dim=3)
        w_miu = F.normalize(w_miu, dim=3)
        theta1 = torch.matmul(n_alpha, n_miu.unsqueeze(dim=3))
        theta2 = torch.matmul(r_hat, n_miu.unsqueeze(dim=3))
        theta3 = torch.sum(r_hat * n_alpha, dim=3, keepdim=True)
        theta4 = torch.matmul(r_miu, n_miu.unsqueeze(dim=3))
        theta5 = torch.sum(n_alpha * r_hat, dim=3, keepdim=True)
        theta6 = torch.sum(n_alpha * v_miu, dim=3, keepdim=True)
        theta7 = torch.sum(n_alpha * w_miu, dim=3, keepdim=True)
        theta8 = torch.sum(r_miu * torch.cross(n_alpha, n_miu.unsqueeze(dim=2).repeat(1, 1, K, 1), dim=3), dim=3, keepdim=True)
        theta9 = torch.norm(r_miu, dim=3, keepdim=True)
        weightNetInput = torch.cat(
            [theta1, theta2, theta3, theta4, theta5, theta6, theta7, theta8, theta9, localized_xyz], dim=3).contiguous()

        gathered_feat = index_points(feats_x, nei_inds)  # [B, M, K, in_ch]
        # new_feat = gathered_feat.permute(0, 3, 2, 1)
        new_feat = torch.cat([gathered_feat, feat_pe.permute(0, 3, 2, 1).contiguous()], dim=-1).permute(0, 3, 2, 1)

        # import ipdb; ipdb.set_trace()
        guidance_feature = new_feat
        guidance_query = guidance_feature # b c k m
        guidance_key = guidance_feature.max(dim=2, keepdim=True)[0].repeat(1, 1, K, 1)
        guidance_score = self.guidance_weight(guidance_query, guidance_key).unsqueeze(1) # b num_heads k n

        # apply guide to features
        new_feat = (new_feat.view(B, -1, self.num_heads, K, M) * guidance_score).view(B, -1, K, M).contiguous()

        weightNetInput = weightNetInput.permute(0, 3, 2, 1)
        weights = self.weightnet(weightNetInput).contiguous()

        # localized_xyz = localized_xyz.permute(0, 3, 2, 1)
        # weights = self.weightnet(localized_xyz)*nei_inds_mask.permute(0,2,1).unsqueeze(dim=1)

        new_feat = torch.matmul(input=new_feat.permute(0, 3, 1, 2).contiguous(), other=weights.permute(0, 3, 2, 1).contiguous()).view(B, M, -1)
        # new_feat = new_feat/nn_idx_divider.unsqueeze(dim=-1)
        new_feat = self.linear(new_feat)
        if self.cfg.BATCH_NORM:
            new_feat = self.bn_linear(new_feat.permute(0, 2, 1))
            new_feat = F.relu(new_feat, inplace=True).permute(0, 2, 1)
        else:
            new_feat = F.relu(new_feat, inplace=True)

        # Dropout
        new_feat = self.dropout(new_feat)

        # Second upscaling mlp
        new_feat = self.unary2(new_feat)

        sparse_feats = torch.max(index_points(dense_feats, nei_inds), dim = 2)[0]

        shortcut = self.unary_shortcut(sparse_feats)

        new_feat = self.leaky_relu(self.drop_path(new_feat) + shortcut)

        return new_feat


class PointConvStridePE(nn.Module):
    def __init__(self, in_channel, out_channel, cfg, weightnet=[9, 16]):
        super(PointConvStridePE, self).__init__()
        self.cfg = cfg
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.drop_path = DropPath(cfg.drop_path_rate) if cfg.drop_path_rate > 0. else nn.Identity()

        # positonal encoder
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_ch = 3
        for out_ch in [out_channel//4, out_channel//4]:
            self.mlp_convs.append(nn.Conv2d(last_ch, out_ch, 1))
            if cfg.BATCH_NORM:
                self.mlp_bns.append(nn.BatchNorm1d(out_ch))
            last_ch = out_ch

        # First downscaling mlp
        if in_channel != out_channel // 4:
            self.unary1 = UnaryBlock(in_channel, out_channel // 4, use_bn=True, bn_momentum=0.1)
        else:
            self.unary1 = nn.Identity()

        self.weightnet = WeightNet(weightnet[0], weightnet[1])
        self.linear = nn.Linear(out_channel // 2 * weightnet[-1], out_channel // 2)
        if cfg.BATCH_NORM:
            self.bn_linear = nn.BatchNorm1d(out_channel // 2)

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

    def forward(self, dense_xyz, sparse_xyz, dense_feats, nei_inds, dense_xyz_norm, sparse_xyz_norm):
        """
        dense_xyz: tensor (batch_size, num_points, 3)
        sparse_xyz: tensor (batch_size, num_points2, 3)
        dense_feats: tensor (batch_size, num_points, num_dims)
        nei_inds: tensor (batch_size, num_points2, K)
        """
        if self.cfg.TIME:
            print("time cut down of PointConvStridePE ************")

        B, N, D = dense_xyz.shape
        _, M, _ = sparse_xyz.shape
        _, _, in_ch = dense_feats.shape
        _, _, K = nei_inds.shape

        # nei_inds = nei_inds.clone().detach()
        # nei_inds_mask = (nei_inds != -1).float()
        # nn_idx_divider = nei_inds_mask.sum(dim = -1)
        # nn_idx_divider[nn_idx_divider == 0] = 1
        # nei_inds[nei_inds == -1] = 0

        # torch.cuda.synchronize()
        t0 = time()
        # First downscaling mlp
        feats_x = self.unary1(dense_feats)

        # torch.cuda.synchronize()
        t1 = time()
        if self.cfg.TIME:
            print("unary1 time: ", t1 - t0)

        gathered_xyz = index_points(dense_xyz, nei_inds)
        # localized_xyz = gathered_xyz - sparse_xyz.view(B, M, 1, D) #[B, M, K, D]
        localized_xyz = gathered_xyz - sparse_xyz.unsqueeze(dim=2)
        gathered_norm = index_points(dense_xyz_norm, nei_inds)

        # torch.cuda.synchronize()
        t2 = time()
        if self.cfg.TIME:
            print("indexing time: ", t2 - t1)

        feat_pe = localized_xyz.permute(0, 3, 2, 1)  # [B, in_ch+D, K, M]
        for i, conv in enumerate(self.mlp_convs):
            if self.cfg.BATCH_NORM:
                bn = self.mlp_bns[i]
                # torch.cuda.synchronize()
                t2 = time()
                feat_pe = conv(feat_pe)
                # torch.cuda.synchronize()
                old_t = t2
                t2 = time()
                if self.cfg.TIME:
                    print("conv time: ", t2 - old_t)
                feat_pe = bn(feat_pe.flatten(2, 3)).reshape_as(feat_pe)
                # torch.cuda.synchronize()
                old_t = t2
                t2 = time()
                if self.cfg.TIME:
                    print("bn time: ", t2 - old_t)
                feat_pe = F.relu(feat_pe, inplace=True)
                # torch.cuda.synchronize()
                old_t = t2
                t2 = time()
                if self.cfg.TIME:
                    print("relu time: ", t2 - old_t)
                #feat_pe = F.relu(bn(conv(feat_pe)), inplace=True)
            else:
                feat_pe = F.relu(conv(feat_pe), inplace=True)

        # torch.cuda.synchronize()
        t3 = time()
        if self.cfg.TIME:
            print("pe time: ", t3 - t2)

        n_alpha = gathered_norm
        n_miu = sparse_xyz_norm
        r_miu = localized_xyz
        r_hat = F.normalize(r_miu, dim=3)
        v_miu = n_miu.unsqueeze(dim=2) - torch.matmul(n_miu.unsqueeze(dim=2), r_hat.permute(0, 1, 3, 2)).permute(0, 1,
                                                                                                                 3,
                                                                                                                 2) * r_hat
        v_miu = F.normalize(v_miu, dim=3)
        w_miu = torch.cross(r_hat, v_miu, dim=3)
        w_miu = F.normalize(w_miu, dim=3)
        theta1 = torch.matmul(n_alpha, n_miu.unsqueeze(dim=3))
        theta2 = torch.matmul(r_hat, n_miu.unsqueeze(dim=3))
        theta3 = torch.sum(r_hat * n_alpha, dim=3, keepdim=True)
        theta4 = torch.matmul(r_miu, n_miu.unsqueeze(dim=3))
        theta5 = torch.sum(n_alpha * r_hat, dim=3, keepdim=True)
        theta6 = torch.sum(n_alpha * v_miu, dim=3, keepdim=True)
        theta7 = torch.sum(n_alpha * w_miu, dim=3, keepdim=True)
        theta8 = torch.sum(r_miu * torch.cross(n_alpha, n_miu.unsqueeze(dim=2).repeat(1, 1, K, 1), dim=3), dim=3,
                           keepdim=True)
        theta9 = torch.norm(r_miu, dim=3, keepdim=True)
        weightNetInput = torch.cat(
            [theta1, theta2, theta3, theta4, theta5, theta6, theta7, theta8, theta9, localized_xyz], dim=3).contiguous()

        # torch.cuda.synchronize()
        t4 = time()
        if self.cfg.TIME:
            print("VI Features time: ", t4 - t3)


        gathered_feat = index_points(feats_x, nei_inds)  # [B, M, K, in_ch]
        new_feat = torch.cat([gathered_feat, feat_pe.permute(0, 3, 2, 1).contiguous()], dim=-1).permute(0, 3, 2, 1)
        # new_feat = gathered_feat.permute(0, 3, 2, 1)

        weightNetInput = weightNetInput.permute(0, 3, 2, 1)
        weights = self.weightnet(weightNetInput)

        # torch.cuda.synchronize()
        t5 = time()
        if self.cfg.TIME:
            print("weightnet time: ", t5 - t4)


        new_feat = torch.matmul(input=new_feat.permute(0, 3, 1, 2), other=weights.permute(0, 3, 2, 1)).view(B, M, -1)
        # new_feat = new_feat/nn_idx_divider.unsqueeze(dim=-1)
        new_feat = self.linear(new_feat)
        if self.cfg.BATCH_NORM:
            new_feat = self.bn_linear(new_feat.permute(0, 2, 1))
            new_feat = F.relu(new_feat, inplace=True).permute(0, 2, 1)
        else:
            new_feat = F.relu(new_feat, inplace=True)

        # Dropout
        new_feat = self.dropout(new_feat)


        # Second upscaling mlp
        new_feat = self.unary2(new_feat)

        sparse_feats = torch.max(index_points(dense_feats, nei_inds), dim = 2)[0]

        shortcut = self.unary_shortcut(sparse_feats)

        new_feat = self.leaky_relu(self.drop_path(new_feat) + shortcut)

        # torch.cuda.synchronize()
        t6 = time()
        if self.cfg.TIME:
            print("pointconv[matmul, linear,...] time: ", t6 - t5)


        return new_feat


class PointConv(nn.Module):
    def __init__(self, in_channel, out_channel, mlp, cfg, weightnet=[9, 16]):
        super(PointConv, self).__init__()
        self.cfg = cfg
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_ch = in_channel + 3
        for out_ch in mlp:
            self.mlp_convs.append(nn.Conv2d(last_ch, out_ch, 1))
            if cfg.BATCH_NORM:
                self.mlp_bns.append(nn.BatchNorm2d(out_ch))
            last_ch = out_ch

        self.weightnet = WeightNet(weightnet[0], weightnet[1])
        self.linear = nn.Linear(last_ch * weightnet[-1], out_channel)
        if cfg.BATCH_NORM:
            self.bn_linear = nn.BatchNorm1d(out_channel)

        self.dropout = nn.Dropout(p=cfg.dropout_rate) if cfg.dropout_rate > 0. else nn.Identity()

    def forward(self, dense_xyz, sparse_xyz, dense_feats, nei_inds, dense_xyz_norm, sparse_xyz_norm):
        """
        dense_xyz: tensor (batch_size, num_points, 3)
        sparse_xyz: tensor (batch_size, num_points2, 3)
        dense_feats: tensor (batch_size, num_points, num_dims)
        nei_inds: tensor (batch_size, num_points2, K)
        """
        B, N, D = dense_xyz.shape
        _, M, _ = sparse_xyz.shape
        _, _, in_ch = dense_feats.shape
        _, _, K = nei_inds.shape

        # nei_inds = nei_inds.clone().detach()
        # nei_inds_mask = (nei_inds != -1).float()
        # nn_idx_divider = nei_inds_mask.sum(dim = -1)
        # nn_idx_divider[nn_idx_divider == 0] = 1
        # nei_inds[nei_inds == -1] = 0

        gathered_xyz = index_points(dense_xyz, nei_inds)
        # localized_xyz = gathered_xyz - sparse_xyz.view(B, M, 1, D) #[B, M, K, D]
        localized_xyz = gathered_xyz - sparse_xyz.unsqueeze(dim=2)
        gathered_norm = index_points(dense_xyz_norm, nei_inds)

        n_alpha = gathered_norm
        n_miu = sparse_xyz_norm
        r_miu = localized_xyz
        r_hat = F.normalize(r_miu, dim=3)
        v_miu = n_miu.unsqueeze(dim=2) - torch.matmul(n_miu.unsqueeze(dim=2), r_hat.permute(0, 1, 3, 2)).permute(0, 1,
                                                                                                                 3,
                                                                                                                 2) * r_hat
        v_miu = F.normalize(v_miu, dim=3)
        w_miu = torch.cross(r_hat, v_miu, dim=3)
        w_miu = F.normalize(w_miu, dim=3)
        theta1 = torch.matmul(n_alpha, n_miu.unsqueeze(dim=3))
        theta2 = torch.matmul(r_hat, n_miu.unsqueeze(dim=3))
        theta3 = torch.sum(r_hat * n_alpha, dim=3, keepdim=True)
        theta4 = torch.matmul(r_miu, n_miu.unsqueeze(dim=3))
        theta5 = torch.sum(n_alpha * r_hat, dim=3, keepdim=True)
        theta6 = torch.sum(n_alpha * v_miu, dim=3, keepdim=True)
        theta7 = torch.sum(n_alpha * w_miu, dim=3, keepdim=True)
        theta8 = torch.sum(r_miu * torch.cross(n_alpha, n_miu.unsqueeze(dim=2).repeat(1, 1, K, 1), dim=3), dim=3,
                           keepdim=True)
        theta9 = torch.norm(r_miu, dim=3, keepdim=True)
        weightNetInput = torch.cat(
            [theta1, theta2, theta3, theta4, theta5, theta6, theta7, theta8, theta9, localized_xyz], dim=3).contiguous()

        gathered_feat = index_points(dense_feats, nei_inds)  # [B, M, K, in_ch]
        new_feat = torch.cat([gathered_feat, localized_xyz], dim=-1)

        new_feat = new_feat.permute(0, 3, 2, 1)  # [B, in_ch+D, K, M]
        for i, conv in enumerate(self.mlp_convs):
            if self.cfg.BATCH_NORM:
                bn = self.mlp_bns[i]
                new_feat = F.relu(bn(conv(new_feat)), inplace=True)
            else:
                new_feat = F.relu(conv(new_feat), inplace=True)

        weightNetInput = weightNetInput.permute(0, 3, 2, 1)
        weights = self.weightnet(weightNetInput)

        # localized_xyz = localized_xyz.permute(0, 3, 2, 1)
        # weights = self.weightnet(localized_xyz)*nei_inds_mask.permute(0,2,1).unsqueeze(dim=1)

        new_feat = torch.matmul(input=new_feat.permute(0, 3, 1, 2), other=weights.permute(0, 3, 2, 1)).view(B, M, -1)
        # new_feat = new_feat/nn_idx_divider.unsqueeze(dim=-1)
        new_feat = self.linear(new_feat)
        if self.cfg.BATCH_NORM:
            new_feat = self.bn_linear(new_feat.permute(0, 2, 1))
            new_feat = F.relu(new_feat, inplace=True).permute(0, 2, 1)
        else:
            new_feat = F.relu(new_feat, inplace=True)

        # Dropout
        new_feat = self.dropout(new_feat)

        return new_feat


class PointConvResBlockGuidedPENewAfter(nn.Module):
    def __init__(self, in_channel, cfg, weightnet=[9, 16], num_heads=4):
        super(PointConvResBlockGuidedPENewAfter, self).__init__()
        self.cfg = cfg
        self.in_channel = in_channel
        self.out_channel = in_channel
        out_channel = in_channel
        self.num_heads = num_heads

        self.drop_path = DropPath(cfg.drop_path_rate) if cfg.drop_path_rate > 0. else nn.Identity()

        # positonal encoder
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_ch = 3
        for out_ch in [out_channel//4, out_channel//4]:
            self.mlp_convs.append(nn.Conv2d(last_ch, out_ch, 1))
            if cfg.BATCH_NORM:
                self.mlp_bns.append(nn.BatchNorm2d(out_ch))
            last_ch = out_ch

        # First downscaling mlp
        if in_channel != out_channel // 4:
            self.unary1 = UnaryBlock(in_channel, out_channel // 4, use_bn=True, bn_momentum=0.1)
        else:
            self.unary1 = nn.Identity()

        assert (out_channel // 2) % num_heads == 0
        self.guidance_weight = MultiHeadGuidanceNewV1(cfg, num_heads, out_channel // 2)
        self.mix_linear = nn.Conv2d(out_channel // 2, out_channel // 2, 1)
    
        self.weightnet = WeightNet(weightnet[0], weightnet[1])
        self.linear = nn.Linear(out_channel // 2 * weightnet[-1], out_channel // 2)
        if cfg.BATCH_NORM:
            self.bn_linear = nn.BatchNorm1d(out_channel // 2)

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

    def forward(self, xyz, feats, nei_inds, xyz_norm):
        """
        xyz: tensor (batch_size, num_points, 3)
        feats: tensor (batch_size, num_points, num_dims)
        nei_inds: tensor (batch_size, num_points, K)
        """
        # import ipdb; ipdb.set_trace()
        B, N, D = xyz.shape
        M = N
        _, _, in_ch = feats.shape
        _, _, K = nei_inds.shape

        # nei_inds = nei_inds.clone().detach()
        # nei_inds_mask = (nei_inds != -1).float()
        # nn_idx_divider = nei_inds_mask.sum(dim = -1)
        # nn_idx_divider[nn_idx_divider == 0] = 1
        # nei_inds[nei_inds == -1] = 0

        # First downscaling mlp
        feats_x = self.unary1(feats)

        gathered_xyz = index_points(xyz, nei_inds)
        # localized_xyz = gathered_xyz - xyz.view(B, M, 1, D) #[B, M, K, D]
        localized_xyz = gathered_xyz - xyz.unsqueeze(dim=2)
        gathered_norm = index_points(xyz_norm, nei_inds)

        feat_pe = localized_xyz.permute(0, 3, 2, 1)  # [B, in_ch+D, K, M]
        for i, conv in enumerate(self.mlp_convs):
            if self.cfg.BATCH_NORM:
                bn = self.mlp_bns[i]
                feat_pe = F.relu(bn(conv(feat_pe)), inplace=True)
            else:
                feat_pe = F.relu(conv(feat_pe), inplace=True)

        n_alpha = gathered_norm
        n_miu = xyz_norm
        r_miu = localized_xyz
        r_hat = F.normalize(r_miu, dim=3)
        v_miu = n_miu.unsqueeze(dim=2) - torch.matmul(n_miu.unsqueeze(dim=2), r_hat.permute(0, 1, 3, 2)).permute(0, 1,
                                                                                                                 3,
                                                                                                                 2) * r_hat
        v_miu = F.normalize(v_miu, dim=3)
        w_miu = torch.cross(r_hat, v_miu, dim=3)
        w_miu = F.normalize(w_miu, dim=3)
        theta1 = torch.matmul(n_alpha, n_miu.unsqueeze(dim=3))
        theta2 = torch.matmul(r_hat, n_miu.unsqueeze(dim=3))
        theta3 = torch.sum(r_hat * n_alpha, dim=3, keepdim=True)
        theta4 = torch.matmul(r_miu, n_miu.unsqueeze(dim=3))
        theta5 = torch.sum(n_alpha * r_hat, dim=3, keepdim=True)
        theta6 = torch.sum(n_alpha * v_miu, dim=3, keepdim=True)
        theta7 = torch.sum(n_alpha * w_miu, dim=3, keepdim=True)
        theta8 = torch.sum(r_miu * torch.cross(n_alpha, n_miu.unsqueeze(dim=2).repeat(1, 1, K, 1), dim=3), dim=3,
                           keepdim=True)
        theta9 = torch.norm(r_miu, dim=3, keepdim=True)
        weightNetInput = torch.cat(
            [theta1, theta2, theta3, theta4, theta5, theta6, theta7, theta8, theta9, localized_xyz], dim=3).contiguous()

        gathered_feat = index_points(feats_x, nei_inds)  # [B, M, K, in_ch]
        # new_feat = torch.cat([gathered_feat, localized_xyz], dim=-1)
        new_feat = torch.cat([gathered_feat, feat_pe.permute(0, 3, 2, 1).contiguous()], dim=-1).permute(0, 3, 2, 1)
        # new_feat = gathered_feat.permute(0, 3, 2, 1)

        # compute guidance weights
        # import ipdb; ipdb.set_trace()
        guidance_query = new_feat # b c k m
        guidance_key = new_feat[:, :, :1, :].repeat(1, 1, K, 1).contiguous()
        # guidance_key = guidance_query.max(dim=2, keepdim=True)[0].repeat(1, 1, K, 1)
        guidance_score = self.guidance_weight(guidance_query, guidance_key).unsqueeze(1) # b num_heads k n

        # apply guide to features
        new_feat = self.mix_linear(new_feat)
        new_feat = (new_feat.view(B, -1, self.num_heads, K, M) * guidance_score).view(B, -1, K, M).contiguous()

        weightNetInput = weightNetInput.permute(0, 3, 2, 1)
        weights = self.weightnet(weightNetInput).contiguous()

        # localized_xyz = localized_xyz.permute(0, 3, 2, 1) # [B, D, K, M]
        # weights = self.weightnet(localized_xyz)*nei_inds_mask.permute(0,2,1).unsqueeze(dim=1)

        new_feat = torch.matmul(input=new_feat.permute(0, 3, 1, 2), other=weights.permute(0, 3, 2, 1)).view(B, M, -1)
        # new_feat = new_feat/nn_idx_divider.unsqueeze(dim=-1)
        new_feat = self.linear(new_feat)
        if self.cfg.BATCH_NORM:
            new_feat = self.bn_linear(new_feat.permute(0, 2, 1))
            new_feat = F.relu(new_feat, inplace=True).permute(0, 2, 1)
        else:
            new_feat = F.relu(new_feat, inplace=True)
        
        # Dropout
        new_feat = self.dropout(new_feat)

        # Second upscaling mlp
        new_feat = self.unary2(new_feat)

        shortcut = self.unary_shortcut(feats)

        new_feat = self.leaky_relu(self.drop_path(new_feat) + shortcut)

        return new_feat


class PointConvResBlockGuidedPEQK(nn.Module):
    def __init__(self, in_channel, cfg, weightnet=[9, 16], num_heads=4):
        super(PointConvResBlockGuidedPEQK, self).__init__()
        self.cfg = cfg
        self.in_channel = in_channel
        self.out_channel = in_channel
        out_channel = in_channel
        self.num_heads = num_heads

        # positonal encoder
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_ch = 3
        for out_ch in [out_channel//4, out_channel//4]:
            self.mlp_convs.append(nn.Conv2d(last_ch, out_ch, 1))
            if cfg.BATCH_NORM:
                self.mlp_bns.append(nn.BatchNorm2d(out_ch))
            last_ch = out_ch

        # First downscaling mlp
        if in_channel != out_channel // 4:
            self.unary1 = UnaryBlock(in_channel, out_channel // 4, use_bn=True, bn_momentum=0.1)
        else:
            self.unary1 = nn.Identity()

        assert (out_channel // 4) % num_heads == 0
        self.attned_feat = MultiHeadGuidanceQK(cfg, num_heads, out_channel // 4, key_dim=16)

        self.mix_linear = nn.Linear(out_channel // 2, out_channel // 2)
    
        self.weightnet = WeightNet(weightnet[0], weightnet[1])
        self.linear = nn.Linear(out_channel // 2 * weightnet[-1], out_channel // 2)
        if cfg.BATCH_NORM:
            self.bn_linear = nn.BatchNorm1d(out_channel // 2)

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

    def forward(self, xyz, feats, nei_inds, xyz_norm):
        """
        xyz: tensor (batch_size, num_points, 3)
        feats: tensor (batch_size, num_points, num_dims)
        nei_inds: tensor (batch_size, num_points, K)
        """

        # import ipdb; ipdb.set_trace()
        B, N, D = xyz.shape
        M = N
        _, _, in_ch = feats.shape
        _, _, K = nei_inds.shape

        # nei_inds = nei_inds.clone().detach()
        # nei_inds_mask = (nei_inds != -1).float()
        # nn_idx_divider = nei_inds_mask.sum(dim = -1)
        # nn_idx_divider[nn_idx_divider == 0] = 1
        # nei_inds[nei_inds == -1] = 0

        # First downscaling mlp
        feats_x = self.unary1(feats)

        gathered_xyz = index_points(xyz, nei_inds)
        # localized_xyz = gathered_xyz - xyz.view(B, M, 1, D) #[B, M, K, D]
        localized_xyz = gathered_xyz - xyz.unsqueeze(dim=2)
        gathered_norm = index_points(xyz_norm, nei_inds)

        feat_pe = localized_xyz.permute(0, 3, 2, 1)  # [B, in_ch+D, K, M]
        for i, conv in enumerate(self.mlp_convs):
            if self.cfg.BATCH_NORM:
                bn = self.mlp_bns[i]
                feat_pe = F.relu(bn(conv(feat_pe)), inplace=True)
            else:
                feat_pe = F.relu(conv(feat_pe), inplace=True)

        n_alpha = gathered_norm
        n_miu = xyz_norm
        r_miu = localized_xyz
        r_hat = F.normalize(r_miu, dim=3)
        v_miu = n_miu.unsqueeze(dim=2) - torch.matmul(n_miu.unsqueeze(dim=2), r_hat.permute(0, 1, 3, 2)).permute(0, 1,
                                                                                                                 3,
                                                                                                                 2) * r_hat
        v_miu = F.normalize(v_miu, dim=3)
        w_miu = torch.cross(r_hat, v_miu, dim=3)
        w_miu = F.normalize(w_miu, dim=3)
        theta1 = torch.matmul(n_alpha, n_miu.unsqueeze(dim=3))
        theta2 = torch.matmul(r_hat, n_miu.unsqueeze(dim=3))
        theta3 = torch.sum(r_hat * n_alpha, dim=3, keepdim=True)
        theta4 = torch.matmul(r_miu, n_miu.unsqueeze(dim=3))
        theta5 = torch.sum(n_alpha * r_hat, dim=3, keepdim=True)
        theta6 = torch.sum(n_alpha * v_miu, dim=3, keepdim=True)
        theta7 = torch.sum(n_alpha * w_miu, dim=3, keepdim=True)
        theta8 = torch.sum(r_miu * torch.cross(n_alpha, n_miu.unsqueeze(dim=2).repeat(1, 1, K, 1), dim=3), dim=3,
                           keepdim=True)
        theta9 = torch.norm(r_miu, dim=3, keepdim=True)
        weightNetInput = torch.cat(
            [theta1, theta2, theta3, theta4, theta5, theta6, theta7, theta8, theta9, localized_xyz], dim=3).contiguous()

        # get attned v
        new_feat = self.attned_feat(feats_x, nei_inds) # B, N, K, in_ch

        new_feat = torch.cat([new_feat, feat_pe.permute(0, 3, 2, 1).contiguous()], dim=-1)
        new_feat = self.mix_linear(new_feat)

        weightNetInput = weightNetInput.permute(0, 3, 2, 1)
        weights = self.weightnet(weightNetInput).contiguous()

        # localized_xyz = localized_xyz.permute(0, 3, 2, 1) # [B, D, K, M]
        # weights = self.weightnet(localized_xyz)*nei_inds_mask.permute(0,2,1).unsqueeze(dim=1)

        new_feat = torch.matmul(input=new_feat.permute(0, 1, 3, 2), other=weights.permute(0, 3, 2, 1)).view(B, M, -1)
        # new_feat = new_feat/nn_idx_divider.unsqueeze(dim=-1)
        new_feat = self.linear(new_feat)
        if self.cfg.BATCH_NORM:
            new_feat = self.bn_linear(new_feat.permute(0, 2, 1))
            new_feat = F.relu(new_feat, inplace=True).permute(0, 2, 1)
        else:
            new_feat = F.relu(new_feat, inplace=True)

        # Second upscaling mlp
        new_feat = self.unary2(new_feat)

        shortcut = self.unary_shortcut(feats)

        new_feat = self.leaky_relu(new_feat + shortcut)

        return new_feat



class PointConvResBlockGuidedPENewAfterVI(nn.Module):
    def __init__(self, in_channel, cfg, weightnet=[9, 16], num_heads=4):
        super(PointConvResBlockGuidedPENewAfterVI, self).__init__()
        self.cfg = cfg
        self.in_channel = in_channel
        self.out_channel = in_channel
        out_channel = in_channel
        self.num_heads = num_heads

        self.drop_path = DropPath(cfg.drop_path_rate) if cfg.drop_path_rate > 0. else nn.Identity()

        # positonal encoder
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_ch = weightnet[0]
        for out_ch in [out_channel//4, out_channel//4]:
            self.mlp_convs.append(nn.Conv2d(last_ch, out_ch, 1))
            if cfg.BATCH_NORM:
                self.mlp_bns.append(nn.BatchNorm2d(out_ch))
            last_ch = out_ch

        # First downscaling mlp
        if in_channel != out_channel // 4:
            self.unary1 = UnaryBlock(in_channel, out_channel // 4, use_bn=True, bn_momentum=0.1)
        else:
            self.unary1 = nn.Identity()

        assert (out_channel // 2) % num_heads == 0
        self.guidance_weight = MultiHeadGuidanceNewV1VI(cfg, num_heads, out_channel // 2)
        self.mix_linear = nn.Conv2d(out_channel // 2, out_channel // 2, 1)
    
        self.weightnet = WeightNet(weightnet[0], weightnet[1])
        self.linear = nn.Linear(out_channel // 2 * weightnet[-1], out_channel // 2)
        if cfg.BATCH_NORM:
            self.bn_linear = nn.BatchNorm1d(out_channel // 2)

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

    def forward(self, xyz, feats, nei_inds, xyz_norm):
        """
        xyz: tensor (batch_size, num_points, 3)
        feats: tensor (batch_size, num_points, num_dims)
        nei_inds: tensor (batch_size, num_points, K)
        """
        # import ipdb; ipdb.set_trace()
        B, N, D = xyz.shape
        M = N
        _, _, in_ch = feats.shape
        _, _, K = nei_inds.shape

        # nei_inds = nei_inds.clone().detach()
        # nei_inds_mask = (nei_inds != -1).float()
        # nn_idx_divider = nei_inds_mask.sum(dim = -1)
        # nn_idx_divider[nn_idx_divider == 0] = 1
        # nei_inds[nei_inds == -1] = 0

        # First downscaling mlp
        feats_x = self.unary1(feats)

        gathered_xyz = index_points(xyz, nei_inds)
        # localized_xyz = gathered_xyz - xyz.view(B, M, 1, D) #[B, M, K, D]
        localized_xyz = gathered_xyz - xyz.unsqueeze(dim=2)
        gathered_norm = index_points(xyz_norm, nei_inds)

        n_alpha = gathered_norm
        n_miu = xyz_norm
        r_miu = localized_xyz
        r_hat = F.normalize(r_miu, dim=3)
        v_miu = n_miu.unsqueeze(dim=2) - torch.matmul(n_miu.unsqueeze(dim=2), r_hat.permute(0, 1, 3, 2)).permute(0, 1,
                                                                                                                 3,
                                                                                                                 2) * r_hat
        v_miu = F.normalize(v_miu, dim=3)
        w_miu = torch.cross(r_hat, v_miu, dim=3)
        w_miu = F.normalize(w_miu, dim=3)
        theta1 = torch.matmul(n_alpha, n_miu.unsqueeze(dim=3))
        theta2 = torch.matmul(r_hat, n_miu.unsqueeze(dim=3))
        theta3 = torch.sum(r_hat * n_alpha, dim=3, keepdim=True)
        theta4 = torch.matmul(r_miu, n_miu.unsqueeze(dim=3))
        theta5 = torch.sum(n_alpha * r_hat, dim=3, keepdim=True)
        theta6 = torch.sum(n_alpha * v_miu, dim=3, keepdim=True)
        theta7 = torch.sum(n_alpha * w_miu, dim=3, keepdim=True)
        theta8 = torch.sum(r_miu * torch.cross(n_alpha, n_miu.unsqueeze(dim=2).repeat(1, 1, K, 1), dim=3), dim=3,
                           keepdim=True)
        theta9 = torch.norm(r_miu, dim=3, keepdim=True)
        weightNetInput = torch.cat(
            [theta1, theta2, theta3, theta4, theta5, theta6, theta7, theta8, theta9, localized_xyz], dim=3).contiguous()


        feat_pe = weightNetInput.permute(0, 3, 2, 1)  # [B, in_ch+D, K, M]
        for i, conv in enumerate(self.mlp_convs):
            if self.cfg.BATCH_NORM:
                bn = self.mlp_bns[i]
                feat_pe = F.relu(bn(conv(feat_pe)), inplace=True)
            else:
                feat_pe = F.relu(conv(feat_pe), inplace=True)

        gathered_feat = index_points(feats_x, nei_inds)  # [B, M, K, in_ch]
        # new_feat = torch.cat([gathered_feat, localized_xyz], dim=-1)
        new_feat = torch.cat([gathered_feat, feat_pe.permute(0, 3, 2, 1).contiguous()], dim=-1).permute(0, 3, 2, 1)
        # new_feat = gathered_feat.permute(0, 3, 2, 1)

        # compute guidance weights
        # import ipdb; ipdb.set_trace()
        guidance_query = gathered_feat.permute(0, 3, 2, 1) # b c k m
        guidance_key = guidance_query[:, :, :1, :].repeat(1, 1, K, 1).contiguous()
        # guidance_key = guidance_query.max(dim=2, keepdim=True)[0].repeat(1, 1, K, 1)
        guidance_score = self.guidance_weight(guidance_query, guidance_key, pos_encoding=feat_pe).unsqueeze(1) # b num_heads k n

        # apply guide to features
        new_feat = self.mix_linear(new_feat)
        new_feat = (new_feat.view(B, -1, self.num_heads, K, M) * guidance_score).view(B, -1, K, M).contiguous()

        weightNetInput = weightNetInput.permute(0, 3, 2, 1)
        weights = self.weightnet(weightNetInput).contiguous()

        # localized_xyz = localized_xyz.permute(0, 3, 2, 1) # [B, D, K, M]
        # weights = self.weightnet(localized_xyz)*nei_inds_mask.permute(0,2,1).unsqueeze(dim=1)

        new_feat = torch.matmul(input=new_feat.permute(0, 3, 1, 2), other=weights.permute(0, 3, 2, 1)).view(B, M, -1)
        # new_feat = new_feat/nn_idx_divider.unsqueeze(dim=-1)
        new_feat = self.linear(new_feat)
        if self.cfg.BATCH_NORM:
            new_feat = self.bn_linear(new_feat.permute(0, 2, 1))
            new_feat = F.relu(new_feat, inplace=True).permute(0, 2, 1)
        else:
            new_feat = F.relu(new_feat, inplace=True)
        
        # Dropout
        new_feat = self.dropout(new_feat)

        # Second upscaling mlp
        new_feat = self.unary2(new_feat)

        shortcut = self.unary_shortcut(feats)

        new_feat = self.leaky_relu(self.drop_path(new_feat) + shortcut)

        return new_feat


class PointConvResBlockGuidedPE(nn.Module):
    def __init__(self, in_channel, cfg, weightnet=[9, 16], num_heads=4):
        super(PointConvResBlockGuidedPE, self).__init__()
        self.cfg = cfg
        self.in_channel = in_channel
        self.out_channel = in_channel
        out_channel = in_channel
        self.num_heads = num_heads

        self.drop_path = DropPath(cfg.drop_path_rate) if cfg.drop_path_rate > 0. else nn.Identity()

        # positonal encoder
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_ch = 3
        for out_ch in [out_channel//4, out_channel//4]:
            self.mlp_convs.append(nn.Conv2d(last_ch, out_ch, 1))
            if cfg.BATCH_NORM:
                self.mlp_bns.append(nn.BatchNorm2d(out_ch))
            last_ch = out_ch

        # First downscaling mlp
        if in_channel != out_channel // 4:
            self.unary1 = UnaryBlock(in_channel, out_channel // 4, use_bn=True, bn_momentum=0.1)
        else:
            self.unary1 = nn.Identity()

        assert (out_channel // 2) % num_heads == 0
        self.guidance_weight = MultiHeadGuidanceNewV1(cfg, num_heads, out_channel // 2)

        self.weightnet = WeightNet(weightnet[0], weightnet[1])
        self.linear = nn.Linear(out_channel // 2 * weightnet[-1], out_channel // 2)
        if cfg.BATCH_NORM:
            self.bn_linear = nn.BatchNorm1d(out_channel // 2)

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

    def forward(self, xyz, feats, nei_inds, xyz_norm):
        """
        xyz: tensor (batch_size, num_points, 3)
        feats: tensor (batch_size, num_points, num_dims)
        nei_inds: tensor (batch_size, num_points, K)
        """
        B, N, D = xyz.shape
        M = N
        _, _, in_ch = feats.shape
        _, _, K = nei_inds.shape

        # nei_inds = nei_inds.clone().detach()
        # nei_inds_mask = (nei_inds != -1).float()
        # nn_idx_divider = nei_inds_mask.sum(dim = -1)
        # nn_idx_divider[nn_idx_divider == 0] = 1
        # nei_inds[nei_inds == -1] = 0

        # First downscaling mlp
        feats_x = self.unary1(feats)

        gathered_xyz = index_points(xyz, nei_inds)
        # localized_xyz = gathered_xyz - xyz.view(B, M, 1, D) #[B, M, K, D]
        localized_xyz = gathered_xyz - xyz.unsqueeze(dim=2)
        gathered_norm = index_points(xyz_norm, nei_inds)

        feat_pe = localized_xyz.permute(0, 3, 2, 1)  # [B, in_ch+D, K, M]
        for i, conv in enumerate(self.mlp_convs):
            if self.cfg.BATCH_NORM:
                bn = self.mlp_bns[i]
                feat_pe = F.relu(bn(conv(feat_pe)), inplace=True)
            else:
                feat_pe = F.relu(conv(feat_pe), inplace=True)

        n_alpha = gathered_norm
        n_miu = xyz_norm
        r_miu = localized_xyz
        r_hat = F.normalize(r_miu, dim=3)
        v_miu = n_miu.unsqueeze(dim=2) - torch.matmul(n_miu.unsqueeze(dim=2), r_hat.permute(0, 1, 3, 2)).permute(0, 1,
                                                                                                                 3,
                                                                                                                 2) * r_hat
        v_miu = F.normalize(v_miu, dim=3)
        w_miu = torch.cross(r_hat, v_miu, dim=3)
        w_miu = F.normalize(w_miu, dim=3)
        theta1 = torch.matmul(n_alpha, n_miu.unsqueeze(dim=3))
        theta2 = torch.matmul(r_hat, n_miu.unsqueeze(dim=3))
        theta3 = torch.sum(r_hat * n_alpha, dim=3, keepdim=True)
        theta4 = torch.matmul(r_miu, n_miu.unsqueeze(dim=3))
        theta5 = torch.sum(n_alpha * r_hat, dim=3, keepdim=True)
        theta6 = torch.sum(n_alpha * v_miu, dim=3, keepdim=True)
        theta7 = torch.sum(n_alpha * w_miu, dim=3, keepdim=True)
        theta8 = torch.sum(r_miu * torch.cross(n_alpha, n_miu.unsqueeze(dim=2).repeat(1, 1, K, 1), dim=3), dim=3,
                           keepdim=True)
        theta9 = torch.norm(r_miu, dim=3, keepdim=True)
        weightNetInput = torch.cat(
            [theta1, theta2, theta3, theta4, theta5, theta6, theta7, theta8, theta9, localized_xyz], dim=3).contiguous()

        gathered_feat = index_points(feats_x, nei_inds)  # [B, M, K, in_ch]
        # new_feat = torch.cat([gathered_feat, localized_xyz], dim=-1)
        new_feat = torch.cat([gathered_feat, feat_pe.permute(0, 3, 2, 1).contiguous()], dim=-1).permute(0, 3, 2, 1)
        # new_feat = gathered_feat.permute(0, 3, 2, 1)

        # compute guidance weights
        # import ipdb; ipdb.set_trace()
        guidance_query = new_feat # b c k m
        guidance_key = new_feat[:, :, :1, :].repeat(1, 1, K, 1).contiguous()
        guidance_score = self.guidance_weight(guidance_query, guidance_key).unsqueeze(1) # b num_heads k n

        # apply guide to features
        new_feat = (new_feat.view(B, -1, self.num_heads, K, M) * guidance_score).view(B, -1, K, M).contiguous()

        weightNetInput = weightNetInput.permute(0, 3, 2, 1)
        weights = self.weightnet(weightNetInput).contiguous()

        # localized_xyz = localized_xyz.permute(0, 3, 2, 1) # [B, D, K, M]
        # weights = self.weightnet(localized_xyz)*nei_inds_mask.permute(0,2,1).unsqueeze(dim=1)

        new_feat = torch.matmul(input=new_feat.permute(0, 3, 1, 2), other=weights.permute(0, 3, 2, 1)).view(B, M, -1)
        # new_feat = new_feat/nn_idx_divider.unsqueeze(dim=-1)
        new_feat = self.linear(new_feat)
        if self.cfg.BATCH_NORM:
            new_feat = self.bn_linear(new_feat.permute(0, 2, 1))
            new_feat = F.relu(new_feat, inplace=True).permute(0, 2, 1)
        else:
            new_feat = F.relu(new_feat, inplace=True)

        # Dropout
        new_feat = self.dropout(new_feat)

        # Second upscaling mlp
        new_feat = self.unary2(new_feat)

        shortcut = self.unary_shortcut(feats)

        new_feat = self.leaky_relu(self.drop_path(new_feat) + shortcut)

        return new_feat


class PointConvTransposePE(nn.Module):
    def __init__(self, in_channel, out_channel, mlp, cfg, weightnet=[9, 16], mlp2=None):
        super(PointConvTransposePE, self).__init__()
        self.cfg = cfg
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.drop_path = DropPath(cfg.drop_path_rate) if cfg.drop_path_rate > 0. else nn.Identity()

        # positonal encoder
        self.pe_convs = nn.ModuleList()
        self.pe_bns = nn.ModuleList()
        last_ch = 3
        for out_ch in [out_channel//4, out_channel//4]:
            self.pe_convs.append(nn.Conv2d(last_ch, out_ch, 1))
            if cfg.BATCH_NORM:
                self.pe_bns.append(nn.BatchNorm2d(out_ch))
            last_ch = out_ch

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_ch = in_channel
        for out_ch in mlp:
            self.mlp_convs.append(nn.Conv2d(last_ch, out_ch, 1))
            if cfg.BATCH_NORM:
                self.mlp_bns.append(nn.BatchNorm2d(out_ch))
            last_ch = out_ch

        self.weightnet = WeightNet(weightnet[0], weightnet[1])
        self.linear = nn.Linear((last_ch + out_channel//4) * weightnet[-1], out_channel)
        if cfg.BATCH_NORM:
            self.bn_linear = nn.BatchNorm1d(out_channel)

        self.dropout = nn.Dropout(p=cfg.dropout_rate) if cfg.dropout_rate > 0. else nn.Identity()

        self.mlp2_convs = nn.ModuleList()
        self.mlp2_bns = nn.ModuleList()
        if mlp2 is not None:
            for i in range(1, len(mlp2)):
                self.mlp2_convs.append(nn.Conv1d(mlp2[i - 1], mlp2[i], 1))
                if cfg.BATCH_NORM:
                    self.mlp2_bns.append(nn.BatchNorm1d(mlp2[i]))

    def forward(self, sparse_xyz, dense_xyz, sparse_feats, dense_feats, nei_inds, dense_xyz_norm, sparse_xyz_norm):
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

        feat_pe = localized_xyz.permute(0, 3, 2, 1)  # [B, in_ch+D, K, M]
        for i, conv in enumerate(self.pe_convs):
            if self.cfg.BATCH_NORM:
                bn = self.pe_bns[i]
                feat_pe = F.relu(bn(conv(feat_pe)), inplace=True)
            else:
                feat_pe = F.relu(conv(feat_pe), inplace=True)

        n_alpha = gathered_norm
        n_miu = dense_xyz_norm
        r_miu = localized_xyz
        r_hat = F.normalize(r_miu, dim=3)
        v_miu = n_miu.unsqueeze(dim=2) - torch.matmul(n_miu.unsqueeze(dim=2), r_hat.permute(0, 1, 3, 2)).permute(0, 1,
                                                                                                                 3,
                                                                                                                 2) * r_hat
        v_miu = F.normalize(v_miu, dim=3)
        w_miu = torch.cross(r_hat, v_miu, dim=3)
        w_miu = F.normalize(w_miu, dim=3)
        theta1 = torch.matmul(n_alpha, n_miu.unsqueeze(dim=3))
        theta2 = torch.matmul(r_hat, n_miu.unsqueeze(dim=3))
        theta3 = torch.sum(r_hat * n_alpha, dim=3, keepdim=True)
        theta4 = torch.matmul(r_miu, n_miu.unsqueeze(dim=3))
        theta5 = torch.sum(n_alpha * r_hat, dim=3, keepdim=True)
        theta6 = torch.sum(n_alpha * v_miu, dim=3, keepdim=True)
        theta7 = torch.sum(n_alpha * w_miu, dim=3, keepdim=True)
        theta8 = torch.sum(r_miu * torch.cross(n_alpha, n_miu.unsqueeze(dim=2).repeat(1, 1, K, 1), dim=3), dim=3,
                           keepdim=True)
        theta9 = torch.norm(r_miu, dim=3, keepdim=True)
        weightNetInput = torch.cat(
            [theta1, theta2, theta3, theta4, theta5, theta6, theta7, theta8, theta9, localized_xyz], dim=3).contiguous()

        gathered_feat = index_points(sparse_feats, nei_inds)  # [B, M, K, in_ch]
        # print(gathered_feat.shape)
        new_feat = torch.cat([gathered_feat, feat_pe.permute(0, 3, 2, 1).contiguous()], dim=-1)

        new_feat = new_feat.permute(0, 3, 2, 1)  # [B, in_ch+D, K, M]
        for i, conv in enumerate(self.mlp_convs):
            if self.cfg.BATCH_NORM:
                bn = self.mlp_bns[i]
                new_feat = F.relu(bn(conv(new_feat)), inplace=True)
            else:
                new_feat = F.relu(conv(new_feat), inplace=True)

        weightNetInput = weightNetInput.permute(0, 3, 2, 1)
        weights = self.weightnet(weightNetInput)

        # localized_xyz = localized_xyz.permute(0, 3, 2, 1) # [B, D, K, M]
        # weights = self.weightnet(localized_xyz)*nei_inds_mask.permute(0,2,1).unsqueeze(dim=1)
        # C_mid = weights.shape[1]

        new_feat = torch.matmul(input=new_feat.permute(0, 3, 1, 2), other=weights.permute(0, 3, 2, 1)).view(B, M, -1)
        # new_feat = new_feat/nn_idx_divider.unsqueeze(dim=-1)
        new_feat = self.linear(new_feat)
        if self.cfg.BATCH_NORM:
            new_feat = self.bn_linear(new_feat.permute(0, 2, 1))
            new_feat = F.relu(new_feat, inplace=True)  # [B, C, M]
        else:
            new_feat = F.relu(new_feat.permute(0, 2, 1), inplace=True)

        if dense_feats is not None:
            # new_feat = torch.cat([new_feat, dense_feats.permute(0, 2, 1)], dim = 1)
            new_feat = new_feat + dense_feats.permute(0, 2, 1)

        # Dropout
        new_feat = self.dropout(new_feat)

        for i, conv in enumerate(self.mlp2_convs):
            if self.cfg.BATCH_NORM:
                bn = self.mlp2_bns[i]
                new_feat = F.relu(bn(conv(new_feat)), inplace=True)
            else:
                new_feat = F.relu(conv(new_feat), inplace=True)

        new_feat = new_feat.permute(0, 2, 1)

        return new_feat


class PointConvResBlockPE(nn.Module):
    def __init__(self, in_channel, cfg, weightnet=[9, 16]):
        super(PointConvResBlockPE, self).__init__()
        self.cfg = cfg
        self.in_channel = in_channel
        self.out_channel = in_channel
        out_channel = in_channel

        self.drop_path = DropPath(cfg.drop_path_rate) if cfg.drop_path_rate > 0. else nn.Identity()

        # positonal encoder
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_ch = 3
        for out_ch in [out_channel//4, out_channel//4]:
            self.mlp_convs.append(nn.Conv2d(last_ch, out_ch, 1))
            if cfg.BATCH_NORM:
                self.mlp_bns.append(nn.BatchNorm2d(out_ch))
            last_ch = out_ch

        # First downscaling mlp
        if in_channel != out_channel // 4:
            self.unary1 = UnaryBlock(in_channel, out_channel // 4, use_bn=True, bn_momentum=0.1)
        else:
            self.unary1 = nn.Identity()

        self.weightnet = WeightNet(weightnet[0], weightnet[1])
        self.linear = nn.Linear(out_channel // 2 * weightnet[-1], out_channel // 2)
        if cfg.BATCH_NORM:
            self.bn_linear = nn.BatchNorm1d(out_channel // 2)

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


    def forward(self, xyz, feats, nei_inds, xyz_norm):
        """
        xyz: tensor (batch_size, num_points, 3)
        feats: tensor (batch_size, num_points, num_dims)
        nei_inds: tensor (batch_size, num_points, K)
        """
        # import ipdb; ipdb.set_trace()
        B, N, D = xyz.shape
        M = N
        _, _, in_ch = feats.shape
        _, _, K = nei_inds.shape

        # nei_inds = nei_inds.clone().detach()
        # nei_inds_mask = (nei_inds != -1).float()
        # nn_idx_divider = nei_inds_mask.sum(dim = -1)
        # nn_idx_divider[nn_idx_divider == 0] = 1
        # nei_inds[nei_inds == -1] = 0

        # First downscaling mlp
        feats_x = self.unary1(feats)

        gathered_xyz = index_points(xyz, nei_inds)
        # localized_xyz = gathered_xyz - xyz.view(B, M, 1, D) #[B, M, K, D]
        localized_xyz = gathered_xyz - xyz.unsqueeze(dim=2)
        gathered_norm = index_points(xyz_norm, nei_inds)

        feat_pe = localized_xyz.permute(0, 3, 2, 1)  # [B, in_ch+D, K, M]
        for i, conv in enumerate(self.mlp_convs):
            if self.cfg.BATCH_NORM:
                bn = self.mlp_bns[i]
                feat_pe = F.relu(bn(conv(feat_pe)), inplace=True)
            else:
                feat_pe = F.relu(conv(feat_pe), inplace=True)

        n_alpha = gathered_norm
        n_miu = xyz_norm
        r_miu = localized_xyz
        r_hat = F.normalize(r_miu, dim=3)
        v_miu = n_miu.unsqueeze(dim=2) - torch.matmul(n_miu.unsqueeze(dim=2), r_hat.permute(0, 1, 3, 2)).permute(0, 1,
                                                                                                                 3,
                                                                                                                 2) * r_hat
        v_miu = F.normalize(v_miu, dim=3)
        w_miu = torch.cross(r_hat, v_miu, dim=3)
        w_miu = F.normalize(w_miu, dim=3)
        theta1 = torch.matmul(n_alpha, n_miu.unsqueeze(dim=3))
        theta2 = torch.matmul(r_hat, n_miu.unsqueeze(dim=3))
        theta3 = torch.sum(r_hat * n_alpha, dim=3, keepdim=True)
        theta4 = torch.matmul(r_miu, n_miu.unsqueeze(dim=3))
        theta5 = torch.sum(n_alpha * r_hat, dim=3, keepdim=True)
        theta6 = torch.sum(n_alpha * v_miu, dim=3, keepdim=True)
        theta7 = torch.sum(n_alpha * w_miu, dim=3, keepdim=True)
        theta8 = torch.sum(r_miu * torch.cross(n_alpha, n_miu.unsqueeze(dim=2).repeat(1, 1, K, 1), dim=3), dim=3,
                           keepdim=True)
        theta9 = torch.norm(r_miu, dim=3, keepdim=True)
        weightNetInput = torch.cat(
            [theta1, theta2, theta3, theta4, theta5, theta6, theta7, theta8, theta9, localized_xyz], dim=3).contiguous()

        gathered_feat = index_points(feats_x, nei_inds)  # [B, M, K, in_ch]
        new_feat = torch.cat([gathered_feat, feat_pe.permute(0, 3, 2, 1).contiguous()], dim=-1)
        # new_feat = gathered_feat

        weightNetInput = weightNetInput.permute(0, 3, 2, 1)
        weights = self.weightnet(weightNetInput)

        # localized_xyz = localized_xyz.permute(0, 3, 2, 1) # [B, D, K, M]
        # weights = self.weightnet(localized_xyz)*nei_inds_mask.permute(0,2,1).unsqueeze(dim=1)

        new_feat = torch.matmul(input=new_feat.permute(0, 1, 3, 2), other=weights.permute(0, 3, 2, 1)).view(B, M, -1)
        # new_feat = new_feat/nn_idx_divider.unsqueeze(dim=-1)
        new_feat = self.linear(new_feat)
        if self.cfg.BATCH_NORM:
            new_feat = self.bn_linear(new_feat.permute(0, 2, 1))
            new_feat = F.relu(new_feat, inplace=True).permute(0, 2, 1)
        else:
            new_feat = F.relu(new_feat, inplace=True)

        # Dropout
        new_feat = self.dropout(new_feat)

        # Second upscaling mlp
        new_feat = self.unary2(new_feat)

        shortcut = self.unary_shortcut(feats)

        new_feat = self.leaky_relu(self.drop_path(new_feat) + shortcut)

        return new_feat


class VI_PointConvPE(nn.Module):
    def __init__(self, cfg):
        super(VI_PointConvPE, self).__init__()

        self.cfg = cfg
        self.total_level = cfg.num_level
        self.guided_level = cfg.guided_level

        self.input_feat_dim = 6 if cfg.USE_XYZ else 3

        self.relu = torch.nn.ReLU(inplace=True)

        weightnet = [cfg.point_dim+9, 16] # 2 hidden layer

        self.selfpointconv = PointConv(self.input_feat_dim, cfg.base_dim, [32, 64], cfg, weightnet)
        self.selfpointconv_res1 = PointConvResBlockPE(cfg.base_dim, cfg, weightnet)
        self.selfpointconv_res2 = PointConvResBlockPE(cfg.base_dim, cfg, weightnet)

        self.pointconv = nn.ModuleList()
        self.pointconv_res = nn.ModuleList()

        for i in range(1, self.total_level):
            in_ch = cfg.feat_dim[i - 1]
            out_ch = cfg.feat_dim[i]

            self.pointconv.append(PointConvStridePE(in_ch, out_ch, cfg, weightnet))

            if self.cfg.resblocks[i] == 0:
                self.pointconv_res.append(nn.ModuleList([]))
            else:
                res_blocks = nn.ModuleList()
                for _ in range(self.cfg.resblocks[i]):
                    res_blocks.append(PointConvResBlockPE(out_ch, cfg, weightnet))

                self.pointconv_res.append(res_blocks)


        self.pointdeconv = nn.ModuleList()
        self.pointdeconv_res = nn.ModuleList()

        for i in range(self.total_level - 2, -1, -1):
            in_ch = cfg.feat_dim[i + 1]
            out_ch = cfg.feat_dim[i]

            mlp2 = [out_ch, out_ch]
            self.pointdeconv.append(PointConvTransposePE(in_ch, out_ch, [], cfg, weightnet, mlp2))

            if self.cfg.resblocks[i] == 0:
                self.pointdeconv_res.append(nn.ModuleList([]))
            else:
                res_blocks = nn.ModuleList()
                for _ in range(self.cfg.resblocks_back[i]):
                    res_blocks.append(PointConvResBlockPE(out_ch, cfg, weightnet))
                self.pointdeconv_res.append(res_blocks)

        #pointwise_decode
        self.fc1 = nn.Linear(cfg.base_dim, cfg.base_dim)
        self.relu = torch.nn.ReLU(inplace=True)
        self.bn = torch.nn.BatchNorm1d(cfg.base_dim)
        self.dropout_fc = torch.nn.Dropout(p=cfg.dropout_fc) if cfg.dropout_fc > 0. else nn.Identity()
        self.fc2 = nn.Linear(cfg.base_dim, cfg.num_classes)

    def forward(self, features, pointclouds, edges_self, edges_forward, edges_propagate, norms):
        # import ipdb; ipdb.set_trace()
        if self.cfg.TIME:
            print("number_of_points: ", features.shape)

        # torch.cuda.synchronize()
        t0 = time()
        #encode pointwise info
        features = torch.cat([features, pointclouds[0]], -1) if self.cfg.USE_XYZ else features
        pointwise_feat = features

        # torch.cuda.synchronize()
        t1 = time()
        if self.cfg.TIME:
            print("pointwise time: ", t1 - t0)
        # level 1 conv
        pointwise_feat = self.selfpointconv(pointclouds[0], pointclouds[0], pointwise_feat, edges_self[0], \
                                            norms[0], norms[0])
        # torch.cuda.synchronize()
        t2 = time()
        if self.cfg.TIME:
            print("level1 selfpointconv time: ", t2 - t1)

        pointwise_feat = self.selfpointconv_res1(pointclouds[0], pointwise_feat, edges_self[0], norms[0])
        pointwise_feat = self.selfpointconv_res2(pointclouds[0], pointwise_feat, edges_self[0], norms[0])
    
        feat_list = [pointwise_feat]
        for i, pointconv in enumerate(self.pointconv):
            dense_xyz = pointclouds[i]
            sparse_xyz = pointclouds[i + 1]

            dense_xyz_norm = norms[i]
            sparse_xyz_norm = norms[i + 1]

            dense_feat = feat_list[-1]
            nei_inds = edges_forward[i]

            sparse_feat = pointconv(dense_xyz, sparse_xyz, dense_feat, nei_inds, dense_xyz_norm, sparse_xyz_norm)

            for res_block in self.pointconv_res[i]:
                nei_inds = edges_self[i + 1]
                sparse_feat = res_block(sparse_xyz, sparse_feat, nei_inds, sparse_xyz_norm)

            feat_list.append(sparse_feat)
            # print(sparse_feat.shape)
            # torch.cuda.synchronize()
            old_t = t2
            t2 = time()
            if self.cfg.TIME:
                print("resblock ", i," time: ", t2 - old_t)
        

        sparse_feat = feat_list[-1]
        for i, pointdeconv in enumerate(self.pointdeconv):
            cur_level = self.total_level - 2 - i
            sparse_xyz = pointclouds[cur_level + 1]
            dense_xyz = pointclouds[cur_level]
            dense_feat = feat_list[cur_level]
            nei_inds = edges_propagate[cur_level]

            dense_xyz_norm = norms[cur_level]
            sparse_xyz_norm = norms[cur_level + 1]

            sparse_feat = pointdeconv(sparse_xyz, dense_xyz, sparse_feat, dense_feat, nei_inds, dense_xyz_norm, sparse_xyz_norm)

            for res_block in self.pointdeconv_res[i]:
                nei_inds = edges_self[cur_level]
                sparse_feat = res_block(dense_xyz, sparse_feat, nei_inds, dense_xyz_norm)

            feat_list[cur_level] = sparse_feat
            # print(sparse_feat.shape)

            # torch.cuda.synchronize()
            old_t = t2
            t2 = time()
            if self.cfg.TIME:
                print("back resblock ", i," time: ", t2 - old_t)
        

        fc = self.dropout_fc(self.relu(self.bn(self.fc1(sparse_feat).permute(0, 2, 1)))).permute(0, 2, 1)
        fc = self.fc2(fc)

        # torch.cuda.synchronize()
        old_t = t2
        t2 = time()
        if self.cfg.TIME:
            print("last linear time: ", t2 - old_t)
    
        return fc


class VI_PointConvGuidedPEQK(nn.Module):
    def __init__(self, cfg):
        super(VI_PointConvGuidedPEQK, self).__init__()

        self.cfg = cfg
        self.total_level = cfg.num_level
        self.guided_level = cfg.guided_level

        self.input_feat_dim = 6 if cfg.USE_XYZ else 3

        self.relu = torch.nn.ReLU(inplace=True)

        weightnet = [cfg.point_dim+9, 16] # 2 hidden layer

        #pointwise_encode
        self.PPmodel = nn.Sequential(
            nn.BatchNorm1d(self.input_feat_dim),

            nn.Linear(self.input_feat_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, cfg.base_dim)
        )

        self.selfpointconv = PointConvStridePE(cfg.base_dim, cfg.base_dim, cfg, weightnet)

        self.pointconv = nn.ModuleList()
        self.pointconv_res = nn.ModuleList()

        for i in range(1, self.total_level):
            in_ch = cfg.feat_dim[i - 1]
            out_ch = cfg.feat_dim[i]

            self.pointconv.append(PointConvStridePE(in_ch, out_ch, cfg, weightnet))

            if self.cfg.resblocks[i] == 0:
                self.pointconv_res.append(nn.ModuleList([]))
            else:
                res_blocks = nn.ModuleList()
                for _ in range(self.cfg.resblocks[i]):
                    if i <= self.guided_level:
                        res_blocks.append(PointConvResBlockPE(out_ch, cfg, weightnet))
                    else:
                        res_blocks.append(PointConvResBlockGuidedPEQK(out_ch, cfg, weightnet, cfg.num_heads))
                self.pointconv_res.append(res_blocks)


        self.pointdeconv = nn.ModuleList()
        self.pointdeconv_res = nn.ModuleList()

        for i in range(self.total_level - 2, -1, -1):
            in_ch = cfg.feat_dim[i + 1]
            out_ch = cfg.feat_dim[i]

            mlp2 = [out_ch, out_ch]
            self.pointdeconv.append(PointConvTransposePE(in_ch, out_ch, [], cfg, weightnet, mlp2))

            if self.cfg.resblocks[i] == 0:
                self.pointdeconv_res.append(nn.ModuleList([]))
            else:
                res_blocks = nn.ModuleList()
                for _ in range(self.cfg.resblocks_back[i]):
                    res_blocks.append(PointConvResBlockPE(out_ch, cfg, weightnet))
                self.pointdeconv_res.append(res_blocks)

        #pointwise_decode
        self.fc1 = nn.Linear(cfg.base_dim, cfg.num_classes)

    def forward(self, features, pointclouds, edges_self, edges_forward, edges_propagate, norms):
        # import ipdb; ipdb.set_trace()

        #encode pointwise info
        features = torch.cat([features, pointclouds[0]], -1) if self.cfg.USE_XYZ else features
        pointwise_feat = self.PPmodel(features[0]).unsqueeze(0)

        # level 1 conv
        pointwise_feat = self.selfpointconv(pointclouds[0], pointclouds[0], pointwise_feat, edges_self[0], \
                                            norms[0], norms[0])

        feat_list = [pointwise_feat]
        for i, pointconv in enumerate(self.pointconv):
            dense_xyz = pointclouds[i]
            sparse_xyz = pointclouds[i + 1]

            dense_xyz_norm = norms[i]
            sparse_xyz_norm = norms[i + 1]

            dense_feat = feat_list[-1]
            nei_inds = edges_forward[i]

            sparse_feat = pointconv(dense_xyz, sparse_xyz, dense_feat, nei_inds, dense_xyz_norm, sparse_xyz_norm)

            for res_block in self.pointconv_res[i]:
                nei_inds = edges_self[i + 1]
                sparse_feat = res_block(sparse_xyz, sparse_feat, nei_inds, sparse_xyz_norm)

            feat_list.append(sparse_feat)
            # print(sparse_feat.shape)

        sparse_feat = feat_list[-1]
        for i, pointdeconv in enumerate(self.pointdeconv):
            cur_level = self.total_level - 2 - i
            sparse_xyz = pointclouds[cur_level + 1]
            dense_xyz = pointclouds[cur_level]
            dense_feat = feat_list[cur_level]
            nei_inds = edges_propagate[cur_level]

            dense_xyz_norm = norms[cur_level]
            sparse_xyz_norm = norms[cur_level + 1]

            sparse_feat = pointdeconv(sparse_xyz, dense_xyz, sparse_feat, dense_feat, nei_inds, dense_xyz_norm, sparse_xyz_norm)

            for res_block in self.pointdeconv_res[i]:
                nei_inds = edges_self[cur_level]
                sparse_feat = res_block(dense_xyz, sparse_feat, nei_inds, dense_xyz_norm)

            feat_list[cur_level] = sparse_feat
            # print(sparse_feat.shape)

        fc = self.fc1(sparse_feat)
        return fc

class VI_PointConvGuidedPEQKV1(nn.Module):
    def __init__(self, cfg):
        super(VI_PointConvGuidedPEQKV1, self).__init__()

        self.cfg = cfg
        self.total_level = cfg.num_level
        self.guided_level = cfg.guided_level

        self.input_feat_dim = 6 if cfg.USE_XYZ else 3

        self.relu = torch.nn.ReLU(inplace=True)

        weightnet = [cfg.point_dim+9, 16] # 2 hidden layer

        #pointwise_encode
        self.selfpointconv = PointConv(self.input_feat_dim, cfg.base_dim, [32, 64], cfg, weightnet)
        self.selfpointconv_res1 = PointConvResBlockPE(cfg.base_dim, cfg, weightnet)
        self.selfpointconv_res2 = PointConvResBlockPE(cfg.base_dim, cfg, weightnet)

        self.pointconv = nn.ModuleList()
        self.pointconv_res = nn.ModuleList()

        for i in range(1, self.total_level):
            in_ch = cfg.feat_dim[i - 1]
            out_ch = cfg.feat_dim[i]

            self.pointconv.append(PointConvStridePE(in_ch, out_ch, cfg, weightnet))

            if self.cfg.resblocks[i] == 0:
                self.pointconv_res.append(nn.ModuleList([]))
            else:
                res_blocks = nn.ModuleList()
                for _ in range(self.cfg.resblocks[i]):
                    if i <= self.guided_level:
                        res_blocks.append(PointConvResBlockPE(out_ch, cfg, weightnet))
                    else:
                        res_blocks.append(PointConvResBlockGuidedPEQK(out_ch, cfg, weightnet, cfg.num_heads))
                self.pointconv_res.append(res_blocks)


        self.pointdeconv = nn.ModuleList()
        self.pointdeconv_res = nn.ModuleList()

        for i in range(self.total_level - 2, -1, -1):
            in_ch = cfg.feat_dim[i + 1]
            out_ch = cfg.feat_dim[i]

            mlp2 = [out_ch, out_ch]
            self.pointdeconv.append(PointConvTransposePE(in_ch, out_ch, [], cfg, weightnet, mlp2))

            if self.cfg.resblocks[i] == 0:
                self.pointdeconv_res.append(nn.ModuleList([]))
            else:
                res_blocks = nn.ModuleList()
                for _ in range(self.cfg.resblocks_back[i]):
                    res_blocks.append(PointConvResBlockPE(out_ch, cfg, weightnet))
                self.pointdeconv_res.append(res_blocks)

        #pointwise_decode
        self.fc1 = nn.Linear(cfg.base_dim, cfg.num_classes)

    def forward(self, features, pointclouds, edges_self, edges_forward, edges_propagate, norms):
        # import ipdb; ipdb.set_trace()

        #encode pointwise info
        features = torch.cat([features, pointclouds[0]], -1) if self.cfg.USE_XYZ else features
        pointwise_feat = features

        # level 1 conv
        pointwise_feat = self.selfpointconv(pointclouds[0], pointclouds[0], pointwise_feat, edges_self[0], \
                                            norms[0], norms[0])
        pointwise_feat = self.selfpointconv_res1(pointclouds[0], pointwise_feat, edges_self[0], norms[0])
        pointwise_feat = self.selfpointconv_res2(pointclouds[0], pointwise_feat, edges_self[0], norms[0])


        feat_list = [pointwise_feat]
        for i, pointconv in enumerate(self.pointconv):
            dense_xyz = pointclouds[i]
            sparse_xyz = pointclouds[i + 1]

            dense_xyz_norm = norms[i]
            sparse_xyz_norm = norms[i + 1]

            dense_feat = feat_list[-1]
            nei_inds = edges_forward[i]

            sparse_feat = pointconv(dense_xyz, sparse_xyz, dense_feat, nei_inds, dense_xyz_norm, sparse_xyz_norm)

            for res_block in self.pointconv_res[i]:
                nei_inds = edges_self[i + 1]
                sparse_feat = res_block(sparse_xyz, sparse_feat, nei_inds, sparse_xyz_norm)

            feat_list.append(sparse_feat)
            # print(sparse_feat.shape)

        sparse_feat = feat_list[-1]
        for i, pointdeconv in enumerate(self.pointdeconv):
            cur_level = self.total_level - 2 - i
            sparse_xyz = pointclouds[cur_level + 1]
            dense_xyz = pointclouds[cur_level]
            dense_feat = feat_list[cur_level]
            nei_inds = edges_propagate[cur_level]

            dense_xyz_norm = norms[cur_level]
            sparse_xyz_norm = norms[cur_level + 1]

            sparse_feat = pointdeconv(sparse_xyz, dense_xyz, sparse_feat, dense_feat, nei_inds, dense_xyz_norm, sparse_xyz_norm)

            for res_block in self.pointdeconv_res[i]:
                nei_inds = edges_self[cur_level]
                sparse_feat = res_block(dense_xyz, sparse_feat, nei_inds, dense_xyz_norm)

            feat_list[cur_level] = sparse_feat
            # print(sparse_feat.shape)

        fc = self.fc1(sparse_feat)
        return fc


class VI_PointConvGuidedPENewAfterDropout(nn.Module):
    def __init__(self, cfg):
        super(VI_PointConvGuidedPENewAfterDropout, self).__init__()

        self.cfg = cfg
        self.total_level = cfg.num_level
        self.guided_level = cfg.guided_level

        self.input_feat_dim = 6 if cfg.USE_XYZ else 3

        self.relu = torch.nn.ReLU(inplace=True)

        weightnet = [cfg.point_dim+9, 16] # 2 hidden layer

        self.selfpointconv = PointConv(self.input_feat_dim, cfg.base_dim, [32, 64], cfg, weightnet)
        self.selfpointconv_res1 = PointConvResBlockPE(cfg.base_dim, cfg, weightnet)
        self.selfpointconv_res2 = PointConvResBlockPE(cfg.base_dim, cfg, weightnet)

        self.pointconv = nn.ModuleList()
        self.pointconv_res = nn.ModuleList()

        for i in range(1, self.total_level):
            in_ch = cfg.feat_dim[i - 1]
            out_ch = cfg.feat_dim[i]

            if i <= self.guided_level:
                self.pointconv.append(PointConvStridePE(in_ch, out_ch, cfg, weightnet))
            else:
                self.pointconv.append(GuidedConvStridePENewAfter(in_ch, out_ch, cfg, weightnet, cfg.num_heads))

            if self.cfg.resblocks[i] == 0:
                self.pointconv_res.append(nn.ModuleList([]))
            else:
                res_blocks = nn.ModuleList()
                for _ in range(self.cfg.resblocks[i]):
                    if i <= self.guided_level:
                        res_blocks.append(PointConvResBlockPE(out_ch, cfg, weightnet))
                    else:
                        res_blocks.append(PointConvResBlockGuidedPENewAfter(out_ch, cfg, weightnet, cfg.num_heads))
                self.pointconv_res.append(res_blocks)


        self.pointdeconv = nn.ModuleList()
        self.pointdeconv_res = nn.ModuleList()

        for i in range(self.total_level - 2, -1, -1):
            in_ch = cfg.feat_dim[i + 1]
            out_ch = cfg.feat_dim[i]

            mlp2 = [out_ch, out_ch]
            self.pointdeconv.append(PointConvTransposePE(in_ch, out_ch, [], cfg, weightnet, mlp2))

            if self.cfg.resblocks[i] == 0:
                self.pointdeconv_res.append(nn.ModuleList([]))
            else:
                res_blocks = nn.ModuleList()
                for _ in range(self.cfg.resblocks_back[i]):
                    res_blocks.append(PointConvResBlockPE(out_ch, cfg, weightnet))
                self.pointdeconv_res.append(res_blocks)

        #pointwise_decode
        self.fc1 = nn.Linear(cfg.base_dim, cfg.base_dim)
        self.relu = torch.nn.ReLU(inplace=True)
        self.bn = torch.nn.BatchNorm1d(cfg.base_dim)
        self.dropout_fc = torch.nn.Dropout(p=cfg.dropout_fc) if cfg.dropout_fc > 0. else nn.Identity()
        self.fc2 = nn.Linear(cfg.base_dim, cfg.num_classes)

    def forward(self, features, pointclouds, edges_self, edges_forward, edges_propagate, norms):
        # import ipdb; ipdb.set_trace()
        if self.cfg.TIME:
            print("number_of_points: ", features.shape)

        # torch.cuda.synchronize()
        t0 = time()
        #encode pointwise info
        features = torch.cat([features, pointclouds[0]], -1) if self.cfg.USE_XYZ else features
        pointwise_feat = features

        # torch.cuda.synchronize()
        t1 = time()
        if self.cfg.TIME:
            print("pointwise time: ", t1 - t0)
        # level 1 conv
        pointwise_feat = self.selfpointconv(pointclouds[0], pointclouds[0], pointwise_feat, edges_self[0], \
                                            norms[0], norms[0])
        # torch.cuda.synchronize()
        t2 = time()
        if self.cfg.TIME:
            print("level1 selfpointconv time: ", t2 - t1)

        pointwise_feat = self.selfpointconv_res1(pointclouds[0], pointwise_feat, edges_self[0], norms[0])
        pointwise_feat = self.selfpointconv_res2(pointclouds[0], pointwise_feat, edges_self[0], norms[0])
    
        feat_list = [pointwise_feat]
        for i, pointconv in enumerate(self.pointconv):
            dense_xyz = pointclouds[i]
            sparse_xyz = pointclouds[i + 1]

            dense_xyz_norm = norms[i]
            sparse_xyz_norm = norms[i + 1]

            dense_feat = feat_list[-1]
            nei_inds = edges_forward[i]

            sparse_feat = pointconv(dense_xyz, sparse_xyz, dense_feat, nei_inds, dense_xyz_norm, sparse_xyz_norm)

            for res_block in self.pointconv_res[i]:
                nei_inds = edges_self[i + 1]
                sparse_feat = res_block(sparse_xyz, sparse_feat, nei_inds, sparse_xyz_norm)

            feat_list.append(sparse_feat)
            # print(sparse_feat.shape)
            # torch.cuda.synchronize()
            old_t = t2
            t2 = time()
            if self.cfg.TIME:
                print("resblock ", i," time: ", t2 - old_t)
        

        sparse_feat = feat_list[-1]
        for i, pointdeconv in enumerate(self.pointdeconv):
            cur_level = self.total_level - 2 - i
            sparse_xyz = pointclouds[cur_level + 1]
            dense_xyz = pointclouds[cur_level]
            dense_feat = feat_list[cur_level]
            nei_inds = edges_propagate[cur_level]

            dense_xyz_norm = norms[cur_level]
            sparse_xyz_norm = norms[cur_level + 1]

            sparse_feat = pointdeconv(sparse_xyz, dense_xyz, sparse_feat, dense_feat, nei_inds, dense_xyz_norm, sparse_xyz_norm)

            for res_block in self.pointdeconv_res[i]:
                nei_inds = edges_self[cur_level]
                sparse_feat = res_block(dense_xyz, sparse_feat, nei_inds, dense_xyz_norm)

            feat_list[cur_level] = sparse_feat
            # print(sparse_feat.shape)

            # torch.cuda.synchronize()
            old_t = t2
            t2 = time()
            if self.cfg.TIME:
                print("back resblock ", i," time: ", t2 - old_t)
        

        fc = self.dropout_fc(self.relu(self.bn(self.fc1(sparse_feat).permute(0, 2, 1)))).permute(0, 2, 1)
        fc = self.fc2(fc)

        # torch.cuda.synchronize()
        old_t = t2
        t2 = time()
        if self.cfg.TIME:
            print("last linear time: ", t2 - old_t)
    
        return fc


class VI_PointConvGuidedPENewAfterDropoutVI(nn.Module):
    def __init__(self, cfg):
        super(VI_PointConvGuidedPENewAfterDropoutVI, self).__init__()

        self.cfg = cfg
        self.total_level = cfg.num_level
        self.guided_level = cfg.guided_level

        self.input_feat_dim = 6 if cfg.USE_XYZ else 3

        self.relu = torch.nn.ReLU(inplace=True)

        weightnet = [cfg.point_dim+9, 16] # 2 hidden layer

        self.selfpointconv = PointConv(self.input_feat_dim, cfg.base_dim, [32, 64], cfg, weightnet)
        self.selfpointconv_res1 = PointConvResBlockPE(cfg.base_dim, cfg, weightnet)
        self.selfpointconv_res2 = PointConvResBlockPE(cfg.base_dim, cfg, weightnet)

        self.pointconv = nn.ModuleList()
        self.pointconv_res = nn.ModuleList()

        for i in range(1, self.total_level):
            in_ch = cfg.feat_dim[i - 1]
            out_ch = cfg.feat_dim[i]

            if i <= self.guided_level:
                self.pointconv.append(PointConvStridePE(in_ch, out_ch, cfg, weightnet))
            else:
                self.pointconv.append(GuidedConvStridePENewAfterVI(in_ch, out_ch, cfg, weightnet, cfg.num_heads))

            if self.cfg.resblocks[i] == 0:
                self.pointconv_res.append(nn.ModuleList([]))
            else:
                res_blocks = nn.ModuleList()
                for _ in range(self.cfg.resblocks[i]):
                    if i <= self.guided_level:
                        res_blocks.append(PointConvResBlockPE(out_ch, cfg, weightnet))
                    else:
                        res_blocks.append(PointConvResBlockGuidedPENewAfterVI(out_ch, cfg, weightnet, cfg.num_heads))
                self.pointconv_res.append(res_blocks)


        self.pointdeconv = nn.ModuleList()
        self.pointdeconv_res = nn.ModuleList()

        for i in range(self.total_level - 2, -1, -1):
            in_ch = cfg.feat_dim[i + 1]
            out_ch = cfg.feat_dim[i]

            mlp2 = [out_ch, out_ch]
            self.pointdeconv.append(PointConvTransposePE(in_ch, out_ch, [], cfg, weightnet, mlp2))

            if self.cfg.resblocks[i] == 0:
                self.pointdeconv_res.append(nn.ModuleList([]))
            else:
                res_blocks = nn.ModuleList()
                for _ in range(self.cfg.resblocks_back[i]):
                    res_blocks.append(PointConvResBlockPE(out_ch, cfg, weightnet))
                self.pointdeconv_res.append(res_blocks)

        #pointwise_decode
        self.fc1 = nn.Linear(cfg.base_dim, cfg.base_dim)
        self.relu = torch.nn.ReLU(inplace=True)
        self.bn = torch.nn.BatchNorm1d(cfg.base_dim)
        self.dropout_fc = torch.nn.Dropout(p=cfg.dropout_fc) if cfg.dropout_fc > 0. else nn.Identity()
        self.fc2 = nn.Linear(cfg.base_dim, cfg.num_classes)

    def forward(self, features, pointclouds, edges_self, edges_forward, edges_propagate, norms):
        # import ipdb; ipdb.set_trace()
        if self.cfg.TIME:
            print("number_of_points: ", features.shape)

        # torch.cuda.synchronize()
        t0 = time()
        #encode pointwise info
        features = torch.cat([features, pointclouds[0]], -1) if self.cfg.USE_XYZ else features
        pointwise_feat = features

        # torch.cuda.synchronize()
        t1 = time()
        if self.cfg.TIME:
            print("pointwise time: ", t1 - t0)
        # level 1 conv
        pointwise_feat = self.selfpointconv(pointclouds[0], pointclouds[0], pointwise_feat, edges_self[0], \
                                            norms[0], norms[0])
        # torch.cuda.synchronize()
        t2 = time()
        if self.cfg.TIME:
            print("level1 selfpointconv time: ", t2 - t1)

        pointwise_feat = self.selfpointconv_res1(pointclouds[0], pointwise_feat, edges_self[0], norms[0])
        pointwise_feat = self.selfpointconv_res2(pointclouds[0], pointwise_feat, edges_self[0], norms[0])
    
        feat_list = [pointwise_feat]
        for i, pointconv in enumerate(self.pointconv):
            dense_xyz = pointclouds[i]
            sparse_xyz = pointclouds[i + 1]

            dense_xyz_norm = norms[i]
            sparse_xyz_norm = norms[i + 1]

            dense_feat = feat_list[-1]
            nei_inds = edges_forward[i]

            sparse_feat = pointconv(dense_xyz, sparse_xyz, dense_feat, nei_inds, dense_xyz_norm, sparse_xyz_norm)

            for res_block in self.pointconv_res[i]:
                nei_inds = edges_self[i + 1]
                sparse_feat = res_block(sparse_xyz, sparse_feat, nei_inds, sparse_xyz_norm)

            feat_list.append(sparse_feat)
            # print(sparse_feat.shape)
            # torch.cuda.synchronize()
            old_t = t2
            t2 = time()
            if self.cfg.TIME:
                print("resblock ", i," time: ", t2 - old_t)
        

        sparse_feat = feat_list[-1]
        for i, pointdeconv in enumerate(self.pointdeconv):
            cur_level = self.total_level - 2 - i
            sparse_xyz = pointclouds[cur_level + 1]
            dense_xyz = pointclouds[cur_level]
            dense_feat = feat_list[cur_level]
            nei_inds = edges_propagate[cur_level]

            dense_xyz_norm = norms[cur_level]
            sparse_xyz_norm = norms[cur_level + 1]

            sparse_feat = pointdeconv(sparse_xyz, dense_xyz, sparse_feat, dense_feat, nei_inds, dense_xyz_norm, sparse_xyz_norm)

            for res_block in self.pointdeconv_res[i]:
                nei_inds = edges_self[cur_level]
                sparse_feat = res_block(dense_xyz, sparse_feat, nei_inds, dense_xyz_norm)

            feat_list[cur_level] = sparse_feat
            # print(sparse_feat.shape)

            # torch.cuda.synchronize()
            old_t = t2
            t2 = time()
            if self.cfg.TIME:
                print("back resblock ", i," time: ", t2 - old_t)
        

        fc = self.dropout_fc(self.relu(self.bn(self.fc1(sparse_feat).permute(0, 2, 1)))).permute(0, 2, 1)
        fc = self.fc2(fc)

        # torch.cuda.synchronize()
        old_t = t2
        t2 = time()
        if self.cfg.TIME:
            print("last linear time: ", t2 - old_t)
    
        return fc


# Residual Block with PointConvFormer
class PointConvResBlockGuidedPENewAfter(nn.Module):
    def __init__(self, in_channel, cfg, weightnet=[9, 16], num_heads=4):
        super(PointConvResBlockGuidedPENewAfter, self).__init__()
        self.cfg = cfg
        self.in_channel = in_channel
        self.out_channel = in_channel
        out_channel = in_channel
        self.num_heads = num_heads

        self.drop_path = DropPath(cfg.drop_path_rate) if cfg.drop_path_rate > 0. else nn.Identity()

        # positonal encoder
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_ch = 3
        for out_ch in [out_channel//4, out_channel//4]:
            self.mlp_convs.append(nn.Conv2d(last_ch, out_ch, 1))
            if cfg.BATCH_NORM:
                self.mlp_bns.append(nn.BatchNorm2d(out_ch))
            last_ch = out_ch

        # First downscaling mlp
        if in_channel != out_channel // 4:
            self.unary1 = UnaryBlock(in_channel, out_channel // 4, use_bn=True, bn_momentum=0.1)
        else:
            self.unary1 = nn.Identity()

        assert (out_channel // 2) % num_heads == 0
        self.guidance_weight = MultiHeadGuidanceNewV1(cfg, num_heads, out_channel // 2)
        self.mix_linear = nn.Conv2d(out_channel // 2, out_channel // 2, 1)
    
        self.weightnet = WeightNet(weightnet[0], weightnet[1])
        self.linear = nn.Linear(out_channel // 2 * weightnet[-1], out_channel // 2)
        if cfg.BATCH_NORM:
            self.bn_linear = nn.BatchNorm1d(out_channel // 2)

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

    def forward(self, xyz, feats, nei_inds, xyz_norm):
        """
        xyz: tensor (batch_size, num_points, 3)
        feats: tensor (batch_size, num_points, num_dims)
        nei_inds: tensor (batch_size, num_points, K)
        """
        # import ipdb; ipdb.set_trace()
        B, N, D = xyz.shape
        M = N
        _, _, in_ch = feats.shape
        _, _, K = nei_inds.shape

        # nei_inds = nei_inds.clone().detach()
        # nei_inds_mask = (nei_inds != -1).float()
        # nn_idx_divider = nei_inds_mask.sum(dim = -1)
        # nn_idx_divider[nn_idx_divider == 0] = 1
        # nei_inds[nei_inds == -1] = 0

        # First downscaling mlp
        feats_x = self.unary1(feats)

        gathered_xyz = index_points(xyz, nei_inds)
        # localized_xyz = gathered_xyz - xyz.view(B, M, 1, D) #[B, M, K, D]
        localized_xyz = gathered_xyz - xyz.unsqueeze(dim=2)
        gathered_norm = index_points(xyz_norm, nei_inds)

        feat_pe = localized_xyz.permute(0, 3, 2, 1)  # [B, in_ch+D, K, M]
        for i, conv in enumerate(self.mlp_convs):
            if self.cfg.BATCH_NORM:
                bn = self.mlp_bns[i]
                feat_pe = F.relu(bn(conv(feat_pe)), inplace=True)
            else:
                feat_pe = F.relu(conv(feat_pe), inplace=True)
        weightNetInput = VI_coordinate_transform(localized_xyz, gathered_norm, xyz_norm, K)

        gathered_feat = index_points(feats_x, nei_inds)  # [B, M, K, in_ch]
        # new_feat = torch.cat([gathered_feat, localized_xyz], dim=-1)
        new_feat = torch.cat([gathered_feat, feat_pe.permute(0, 3, 2, 1).contiguous()], dim=-1).permute(0, 3, 2, 1)
        # new_feat = gathered_feat.permute(0, 3, 2, 1)

        # compute guidance weights
        # import ipdb; ipdb.set_trace()
        guidance_query = new_feat # b c k m
        guidance_key = new_feat[:, :, :1, :].repeat(1, 1, K, 1).contiguous()
        # guidance_key = guidance_query.max(dim=2, keepdim=True)[0].repeat(1, 1, K, 1)
        guidance_score = self.guidance_weight(guidance_query, guidance_key).unsqueeze(1) # b num_heads k n

        # apply guide to features
        new_feat = self.mix_linear(new_feat)
        new_feat = (new_feat.view(B, -1, self.num_heads, K, M) * guidance_score).view(B, -1, K, M).contiguous()

        weightNetInput = weightNetInput.permute(0, 3, 2, 1)
        weights = self.weightnet(weightNetInput).contiguous()

        # localized_xyz = localized_xyz.permute(0, 3, 2, 1) # [B, D, K, M]
        # weights = self.weightnet(localized_xyz)*nei_inds_mask.permute(0,2,1).unsqueeze(dim=1)

        new_feat = torch.matmul(input=new_feat.permute(0, 3, 1, 2), other=weights.permute(0, 3, 2, 1)).view(B, M, -1)
        # new_feat = new_feat/nn_idx_divider.unsqueeze(dim=-1)
        new_feat = self.linear(new_feat)
        if self.cfg.BATCH_NORM:
            new_feat = self.bn_linear(new_feat.permute(0, 2, 1))
            new_feat = F.relu(new_feat, inplace=True).permute(0, 2, 1)
        else:
            new_feat = F.relu(new_feat, inplace=True)
        
        # Dropout
        new_feat = self.dropout(new_feat)

        # Second upscaling mlp
        new_feat = self.unary2(new_feat)

        shortcut = self.unary_shortcut(feats)

        new_feat = self.leaky_relu(self.drop_path(new_feat) + shortcut)

        return new_feat


