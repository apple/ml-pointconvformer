#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022-2023 Apple Inc. All Rights Reserved.
#

import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import DropPath

from util.checkpoint import CheckpointFunction
from layer_utils import PConv, PCF, index_points, VI_coordinate_transform, Linear_BN, UnaryBlock

# Main PointConv/PointConvFormer Layers are:
# PointConv, PointConvStridePE, PCFLayer, PointConvTransposePE


# Multi-head Guidance:
# Input: guidance_query: input features (B x N x K x C)
#        guidance_key: also input features (but less features when downsampling)
#        pos_encoding: if not None, then position encoding is concatenated with the features
# Output: guidance_features: (B x N x K x num_heads)
class MultiHeadGuidance(nn.Module):
    """ Multi-head guidance to increase model expressivitiy"""

    def __init__(self, cfg, num_heads: int, num_hiddens: int):
        super().__init__()
        # assert num_hiddens % num_heads == 0, 'num_hiddens: %d, num_heads: %d'%(num_hiddens, num_heads)
        self.cfg = cfg
        self.dim = num_hiddens
        self.num_heads = num_heads

        self.layer_norm_q = nn.LayerNorm(
            num_hiddens) if cfg.layer_norm_guidance else nn.Identity()
        self.layer_norm_k = nn.LayerNorm(
            num_hiddens) if cfg.layer_norm_guidance else nn.Identity()

        self.mlp = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        mlp_dim = [self.dim, 8, num_heads]
        for ch_in, ch_out in zip(mlp_dim[:-1], mlp_dim[1:]):
            if cfg.BATCH_NORM:
                self.mlp.append(Linear_BN(ch_in, ch_out))
            else:
                self.mlp.append(nn.Linear(ch_in, ch_out))

    def forward(self, guidance_query, guidance_key):  # , pos_encoding=None):

        # attention bxnxkxc
        # batch_dim, n, k, _ = guidance_query.shape

        scores = self.layer_norm_q(guidance_query) - \
            self.layer_norm_k(guidance_key)
        # scores = scores if pos_encoding is None else scores + pos_encoding.permute(0, 2, 3, 1)

        for i, layer in enumerate(self.mlp):
            scores = layer(scores)

            if i == len(self.mlp) - 1:
                scores = torch.sigmoid(scores)
                # The following are alternatives to sigmoid, disabled right now since all perform significantly worse
                # scores = torch.nn.functional.softmax(scores, dim = 2)
                # scores = F.relu(scores)
                # scores = torch.tanh(scores).squeeze(-1)
            else:
                scores = F.relu(scores, inplace=True)

        return scores


# Multi-head Guidance using the inner product of QK, as in conventional attention models. However,
# a sigmoid function is used as activation
# Input: guidance_query: input features (B x N x K x C)
#        guidance_key: also input features (but less features when downsampling)
#        pos_encoding: if not None, then position encoding is concatenated with the features
# Output: guidance_features: (B x N x K x num_heads)
class MultiHeadGuidanceQK(nn.Module):
    """ Multi-head guidance to increase model expressivitiy"""

    def __init__(self, cfg, num_heads: int, num_hiddens: int, key_dim: int):
        super().__init__()
        assert num_hiddens % num_heads == 0, 'num_hiddens: %d, num_heads: %d' % (
            num_hiddens, num_heads)
        self.cfg = cfg
        self.dim = num_hiddens
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.scale = self.key_dim ** -0.5
        self.qk_linear = Linear_BN(self.dim, key_dim * num_heads)

    def forward(self, q, k):
        # input q: b, n,k, c
        #       k: b, n,k,c

        # compute q, k
        B, N, K, _ = q.shape

        q = self.qk_linear(q)
        k = self.qk_linear(k)
        q = q.view(B, N, K, self.num_heads, -1)
        k = k.view(B, N, K, self.num_heads, -1)
        # actually there is only one center..
        k = k[:, :, :1, :, :]
        q = q.transpose(2, 3)
        k = k.permute(0, 1, 3, 4, 2)

        # compute attention
        attn_score = (q @ k) * self.scale
        attn_score = attn_score[:, :, :, :, 0].transpose(2, 3)
        # Disabled softmax version since it performs significantly worse
        # attn_score = F.softmax(attn_score, dim = 2)
        attn_score = torch.sigmoid(attn_score)

        return attn_score


def _bn_function_factory(mlp_convs):
    # Used for the gradient checkpointing in WeightNet
    def bn_function(*inputs):
        output = inputs[0]
        for conv in mlp_convs:
            output = F.relu(conv(output), inplace=True)
        return output
    return bn_function


class WeightNet(nn.Module):
    '''
    WeightNet for PointConv. This runs a few MLP layers (defined by hidden_unit) on the 
    point coordinates and outputs generated weights for each neighbor of each point. 
    The weights will then be matrix-multiplied with the input to perform convolution

    Parameters:
        in_channel: Number of input channels
        out_channel: Number of output channels
        hidden_unit: Number of hidden units, a list which can contain multiple hidden layers
        efficient: If set to True, then gradient checkpointing is used in training to reduce memory cost
    Input: Coordinates for all the kNN neighborhoods
           input shape is B x N x K x in_channel, B is batch size, in_channel is the dimensionality of
            the coordinates (usually 3 for 3D or 2 for 2D, 12 for VI), K is the neighborhood size,
            N is the number of points
    Output: The generated weights B x N x K x C_mid
    '''
    def __init__(
            self,
            in_channel,
            out_channel,
            hidden_unit=[8, 8],
            efficient=False):
        super(WeightNet, self).__init__()

        self.mlp_convs = nn.ModuleList()
        self.efficient = efficient
        if hidden_unit is None or len(hidden_unit) == 0:
            self.mlp_convs.append(Linear_BN(in_channel, out_channel))
        else:
            self.mlp_convs.append(Linear_BN(in_channel, hidden_unit[0]))
            for i in range(1, len(hidden_unit)):
                self.mlp_convs.append(
                    Linear_BN(hidden_unit[i - 1], hidden_unit[i]))
            self.mlp_convs.append(Linear_BN(hidden_unit[-1], out_channel))

    def real_forward(self, localized_xyz):
        # xyz : BxNxKxC
        weights = localized_xyz
        for conv in self.mlp_convs:
            weights = conv(weights)
#        if i < len(self.mlp_convs) - 1:
            weights = F.relu(weights, inplace=True)

        return weights

    def forward(self, localized_xyz):
        if self.efficient and self.training:
            # Try this so that weights have gradient
            #            weights = self.mlp_convs[0](localized_xyz)
            conv_bn_relu = _bn_function_factory(self.mlp_convs)
            dummy = torch.zeros(
                1,
                dtype=torch.float32,
                requires_grad=True,
                device=localized_xyz.device)
            args = [localized_xyz + dummy]
            if self.training:
                for conv in self.mlp_convs:
                    args += tuple(conv.bn.parameters())
                    args += tuple(conv.c.parameters())
                weights = CheckpointFunction.apply(conv_bn_relu, 1, *args)
        else:
            weights = self.real_forward(localized_xyz)
        return weights


class PCFLayer(nn.Module):
    '''
    PointConvFormer main layer
    Parameters:
        in_channel: Number of input channels
        out_channel: Number of output channels
        weightnet: Number of input/output channels for weightnet
        num_heads: Number of heads
        guidance_feat_len: Number of dimensions of the query/key features
    Input:
        dense_xyz: tensor (batch_size, num_points, 3). The coordinates of the points before subsampling 
                   (if it is a "strided" convolution wihch simultaneously subsamples the point cloud)
        dense_feats: tensor (batch_size, num_points, num_dims). The features of the points before subsampling.
        nei_inds: tensor (batch_size, num_points2, K). The neighborhood indices of the K nearest neighbors 
                  of each point (after subsampling). The indices should index into dense_xyz and dense_feats,
                  as during subsampling features at new coordinates are aggregated from the points before subsampling
        dense_xyz_norm: tensor (batch_size, num_points, 3). The surface normals of the points before subsampling
        sparse_xyz: tensor (batch_size, num_points2, 3). The coordinates of the points after subsampling (if there 
                    is no subsampling, just input None for this and the next)
        sparse_xyz_norm: tensor (batch_size, num_points2, 3). The surface normals of the points after subsampling
        vi_features: tensor (batch_size, num_points2, 12). VI features only needs to be computed once per stage. If 
                     it has been computed in a previous layer, it can be saved and directly inputted here.
        Note: batch_size is usually 1 since we are using the packed representation packing multiple point clouds into one. However this dimension needs to be there for pyTorch to work properly.
    Output:
        new_feat: output features
        weightNetInput: the input to weightNet, which are relative coordinates or viewpoint-invariance aware transforms of it
    '''
    def __init__(
            self,
            in_channel,
            out_channel,
            cfg,
            weightnet=[
                9,
                16],
            num_heads=4,
            guidance_feat_len=32):
        super(PCFLayer, self).__init__()
        self.cfg = cfg
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.num_heads = num_heads

        self.drop_path = DropPath(
            cfg.drop_path_rate) if cfg.drop_path_rate > 0. else nn.Identity()

        if cfg.BATCH_NORM:
            self.mlp_conv = Linear_BN(12, guidance_feat_len)
        else:
            self.mlp_conv = nn.Linear(12, guidance_feat_len)

        # First downscaling mlp
        if in_channel != out_channel // 4:
            self.unary1 = UnaryBlock(
                in_channel,
                out_channel // 4,
                use_bn=True,
                bn_momentum=0.1)
        else:
            self.unary1 = nn.Identity()

        self.guidance_unary = UnaryBlock(
            out_channel // 4,
            guidance_feat_len,
            use_bn=True,
            bn_momentum=0.1,
            no_relu=True)

        # check last_ch % num_heads == 0
        assert (out_channel // 2) % num_heads == 0
        if cfg.attention_type == 'subtraction':
            self.guidance_weight = MultiHeadGuidance(
                cfg, num_heads, 2 * guidance_feat_len)
        else:
            self.guidance_weight = MultiHeadGuidanceQK(
                cfg, num_heads, 2 * guidance_feat_len, key_dim=16)

        self.weightnet = WeightNet(weightnet[0], weightnet[1], efficient=True)
        if cfg.BATCH_NORM:
            self.linear = Linear_BN(
                out_channel // 4 * weightnet[-1], out_channel // 2, bn_ver='1d')
        else:
            self.linear = nn.Linear(
                out_channel // 4 * weightnet[-1], out_channel // 2)

        self.dropout = nn.Dropout(
            p=cfg.dropout_rate) if cfg.dropout_rate > 0. else nn.Identity()

        # Second upscaling mlp
        self.unary2 = UnaryBlock(
            out_channel // 2,
            out_channel,
            use_bn=True,
            bn_momentum=0.1,
            no_relu=True)

        # Shortcut optional mpl
        if in_channel != out_channel:
            self.unary_shortcut = UnaryBlock(
                in_channel,
                out_channel,
                use_bn=True,
                bn_momentum=0.1,
                no_relu=True)
        else:
            self.unary_shortcut = nn.Identity()

        # Other operations
        self.leaky_relu = nn.LeakyReLU(0.1)

        return

    def forward(
            self,
            dense_xyz,
            dense_feats,
            nei_inds,
            dense_xyz_norm,
            sparse_xyz=None,
            sparse_xyz_norm=None,
            vi_features=None):
        """
        dense_xyz: tensor (batch_size, num_points, 3)
        dense_feats: tensor (batch_size, num_points, num_dims)
        nei_inds: tensor (batch_size, num_points2, K)
        dense_xyz_norm: tensor (batch_size, num_points, 3)
        sparse_xyz: tensor (batch_size, num_points2, 3)
        sparse_xyz_norm: tensor (batch_size, num_points2, 3)
        vi_features: tensor (batch_size, num_points2, 12). VI features only needs to be computed once per stage. If it has been computed in a previous layer,
                     it can be saved and directly inputted here.
        """
        B, N, _ = dense_xyz.shape
        if sparse_xyz is not None:
            _, M, _ = sparse_xyz.shape
        else:
            M = N
        _, _, K = nei_inds.shape
        # first downscaling mlp
        feats_x = self.unary1(dense_feats)

        gathered_xyz = index_points(dense_xyz, nei_inds)
        # localized_xyz = gathered_xyz - sparse_xyz.view(B, M, 1, D) #[B, M, K,
        # D]
        if sparse_xyz is not None:
            localized_xyz = gathered_xyz - sparse_xyz.unsqueeze(dim=2)
        else:
            localized_xyz = gathered_xyz - dense_xyz.unsqueeze(dim=2)
        gathered_norm = index_points(dense_xyz_norm, nei_inds)

        if self.cfg.USE_VI is True:
            if vi_features is None:
                if sparse_xyz is not None:
                    weightNetInput = VI_coordinate_transform(
                        localized_xyz, gathered_norm, sparse_xyz_norm, K)
                else:
                    weightNetInput = VI_coordinate_transform(
                        localized_xyz, gathered_norm, dense_xyz_norm, K)

            else:
                weightNetInput = vi_features
        else:
            weightNetInput = localized_xyz
        # Encode weightNetInput to be higher dimensional to match with gathered
        # feat
        feat_pe = self.mlp_conv(weightNetInput)
        feat_pe = F.relu(feat_pe)

        if not self.cfg.USE_CUDA_KERNEL:
            gathered_feat = index_points(feats_x, nei_inds)
            gathered_feat = gathered_feat.permute(0, 3, 2, 1)
        # First downsample on the feature dimension, so that it matches the
        # position encoding dimension
        guidance_x = self.guidance_unary(feats_x)
        # Gather features on this low dimensionality is faster and uses less
        # memory
        gathered_feat2 = index_points(guidance_x, nei_inds)  # [B, M, K, in_ch]
        # new_feat = gathered_feat.permute(0, 3, 2, 1)
        guidance_feature = torch.cat([gathered_feat2, feat_pe], dim=-1)

        guidance_query = guidance_feature  # b m k c
        if M == N:
            guidance_key = guidance_feature[:, :, :1, :].repeat(1, 1, K, 1)
        else:
            guidance_key = guidance_feature.max(dim=2, keepdim=True)[
                0].repeat(1, 1, K, 1)
        guidance_score = self.guidance_weight(guidance_query, guidance_key)  # b n k num_heads
        # WeightNet computes the convolutional weights
        weights = self.weightnet(weightNetInput)

        if not self.cfg.USE_CUDA_KERNEL:
            gathered_feat = (gathered_feat.view(B, -1, self.num_heads, K, M)
                             * guidance_score.permute(0, 3, 2, 1)).view(B, -1, K, M)
            gathered_feat = torch.matmul(input=gathered_feat.permute(0, 3, 1, 2).contiguous(),
                                         other=weights).view(B, M, -1)
        else:
            gathered_feat = PCF.forward(feats_x.contiguous(), nei_inds.contiguous(), guidance_score.contiguous(), weights.contiguous())
        new_feat = self.linear(gathered_feat)
        new_feat = F.relu(new_feat, inplace=True)

        # Dropout
        new_feat = self.dropout(new_feat)

        # Second upscaling mlp
        new_feat = self.unary2(new_feat)

        # TODO: some speed-up opportunities here to shave a few milliseconds
        if sparse_xyz is not None:
            sparse_feats = torch.max(
                index_points(
                    dense_feats,
                    nei_inds),
                dim=2)[0]
        else:
            sparse_feats = dense_feats

        shortcut = self.unary_shortcut(sparse_feats)

        new_feat = self.leaky_relu(self.drop_path(new_feat) + shortcut)

        return new_feat, weightNetInput


class PointTransformerLayer(nn.Module):
    '''
    PointTransformer layer, provided for ablation, code adapted from https://github.com/POSTECH-CVLab/point-transformer
    Parameters:
        in_planes: Number of input channels
        out_planes: Number of output channels
        shared_planes: Number of heads
    Input:
        xyz: tensor (batch_size, num_points, 3). The coordinates of the points before subsampling (if it is a "strided" 
             convolution wihch simultaneously subsamples the point cloud)
        feats: tensor (batch_size, num_points, num_dims). The features of the points before subsampling.
        nei_inds: tensor (batch_size, num_points2, K). The neighborhood indices of the K nearest neighbors of each point 
                  (after subsampling). The indices should index into dense_xyz and dense_feats,
                  as during subsampling features at new coordinates are aggregated from the points before subsampling
        sparse_xyz: tensor (batch_size, num_points2, 3). The coordinates of the points after subsampling (if there is no 
                    subsampling, just input None for this and the next)
        Note: batch_size is usually 1 since we are using the packed representation packing multiple point clouds into one. 
        However this dimension needs to be there for pyTorch to work properly.
    Output:
        new_feat: output features
    '''
    def __init__(self, in_planes, out_planes, share_planes=8):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)
        self.linear_p = nn.Sequential(
            Linear_BN(
                3, 3, bn_ver='1d'), nn.ReLU(
                inplace=True), nn.Linear(
                3, out_planes))
        self.bn_w = nn.BatchNorm1d(mid_planes)
        self.linear_w = nn.Sequential(
            nn.ReLU(
                inplace=True),
            Linear_BN(
                mid_planes,
                mid_planes //
                share_planes,
                bn_ver='1d'),
            nn.ReLU(
                inplace=True),
            nn.Linear(
                mid_planes //
                share_planes,
                out_planes //
                share_planes))
        self.softmax = nn.Softmax(dim=1)
        if in_planes != out_planes:
            self.unary_shortcut = UnaryBlock(
                in_planes,
                out_planes,
                use_bn=True,
                bn_momentum=0.1,
                no_relu=True)
        else:
            self.unary_shortcut = nn.Identity()
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, xyz, feats, nei_ind, sparse_xyz=None) -> torch.Tensor:
        # xyz: b, n, 3  nei_ind: b, n, k
        # feats: b, n, c
        _, n, _ = feats.shape
        _, _, k = nei_ind.shape
        if sparse_xyz is not None:
            _, M, _ = sparse_xyz.shape
        else:
            M = n

        feats_q, feats_k, feats_v = self.linear_q(
            feats), self.linear_k(feats), self.linear_v(feats)
#        feats_q = feats_q.squeeze(0)
        feats_k = index_points(feats_k, nei_ind).squeeze(0)  # n, k, c_mid
        feats_v = index_points(feats_v, nei_ind).squeeze(0)  # n, k, c_mid
        if sparse_xyz is not None:
            dxyz = (index_points(xyz, nei_ind) - sparse_xyz.unsqueeze(dim=2))
            feats_q = index_points(feats_q, nei_ind[:, :, 0].unsqueeze(dim=2))
        else:
            dxyz = (
                index_points(
                    xyz,
                    nei_ind) -
                xyz.unsqueeze(
                    dim=2))  # n, k, 3
            feats_q = feats_q.unsqueeze(dim=2)
        dxyz = dxyz.squeeze(0)
        for layer in self.linear_p:
            dxyz = layer(dxyz)
        w = feats_k - feats_q[0] + dxyz.view(M,
                                             k,
                                             self.out_planes // self.mid_planes,
                                             self.mid_planes).sum(2)  # n, k, c_mid
        w = w.transpose(1, 2)
        w = self.bn_w(w)
        w = w.transpose(2, 1)
        for layer in self.linear_w:
            w = layer(w)
        w = self.softmax(w)
        c = feats_v.shape[-1]
        s = self.share_planes
        new_feats = (
            (feats_v +
             dxyz).view(
                M,
                k,
                s,
                c //
                s) *
            w.unsqueeze(2)).sum(1).view(
            M,
            c)
        if sparse_xyz is not None:
            sparse_feats = torch.max(index_points(feats, nei_ind), dim=2)[0]
        else:
            sparse_feats = feats
        shortcut = self.unary_shortcut(sparse_feats)
        new_feats = self.leaky_relu(new_feats + shortcut)
        return new_feats


class PointConvStridePE(nn.Module):
    '''
    PointConv layer with a positional embedding concatenated to the features
    Parameters:
        in_channel: Number of input channels
        out_channel: Number of output channels
        weightnet: Number of input/output channels for weightnet
    Input:
        dense_xyz: tensor (batch_size, num_points, 3). The coordinates of the points before subsampling (if it is a "strided" convolution wihch simultaneously subsamples the point cloud)
        dense_feats: tensor (batch_size, num_points, num_dims). The features of the points before subsampling.
        nei_inds: tensor (batch_size, num_points2, K). The neighborhood indices of the K nearest neighbors of each point (after subsampling). The indices should index into dense_xyz and dense_feats,
                  as during subsampling features at new coordinates are aggregated from the points before subsampling
        dense_xyz_norm: tensor (batch_size, num_points, 3). The surface normals of the points before subsampling
        sparse_xyz: tensor (batch_size, num_points2, 3). The coordinates of the points after subsampling (if there is no subsampling, just input None for this and the next)
        sparse_xyz_norm: tensor (batch_size, num_points2, 3). The surface normals of the points after subsampling
        vi_features: tensor (batch_size, num_points2, 12). VI features only needs to be computed once per stage. If it has been computed in a previous layer,
                     it can be saved and directly inputted here.
        Note: batch_size is usually 1 since we are using the packed representation packing multiple point clouds into one. However this dimension needs to be there for pyTorch to work properly.
    Output:
        new_feat: output features
        weightNetInput: the input to weightNet, which are relative coordinates
                        or viewpoint-invariance aware transforms of it
    '''
    def __init__(self, in_channel, out_channel, cfg, weightnet=[9, 16]):
        super(PointConvStridePE, self).__init__()
        self.cfg = cfg
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.drop_path = DropPath(
            cfg.drop_path_rate) if cfg.drop_path_rate > 0. else nn.Identity()

        # positonal encoder
        self.pe_convs = WeightNet(
            3, min(out_channel // 4, 32), hidden_unit=[out_channel // 4], efficient=True)
        last_ch = min(out_channel // 4, 32)

        # First downscaling mlp
        if in_channel != out_channel // 4:
            self.unary1 = UnaryBlock(
                in_channel,
                out_channel // 4,
                use_bn=True,
                bn_momentum=0.1)
        else:
            self.unary1 = nn.Identity()

        self.weightnet = WeightNet(weightnet[0], weightnet[1], efficient=True)
        if cfg.BATCH_NORM:
            self.linear = Linear_BN(
                (out_channel // 4 + last_ch) * weightnet[-1], out_channel // 2, bn_ver='1d')
        else:
            self.linear = nn.Linear(
                (out_channel // 4 + last_ch) * weightnet[-1], out_channel // 2)

        self.dropout = nn.Dropout(
            p=cfg.dropout_rate) if cfg.dropout_rate > 0. else nn.Identity()

        # Second upscaling mlp
        self.unary2 = UnaryBlock(
            out_channel // 2,
            out_channel,
            use_bn=True,
            bn_momentum=0.1,
            no_relu=True)

        # Shortcut optional mpl
        if in_channel != out_channel:
            self.unary_shortcut = UnaryBlock(
                in_channel,
                out_channel,
                use_bn=True,
                bn_momentum=0.1,
                no_relu=True)
        else:
            self.unary_shortcut = nn.Identity()

        # Other operations
        self.leaky_relu = nn.LeakyReLU(0.1)

        return

    def forward(
            self,
            dense_xyz,
            dense_feats,
            nei_inds,
            dense_xyz_norm,
            sparse_xyz=None,
            sparse_xyz_norm=None,
            vi_features=None):
        """
        dense_xyz: tensor (batch_size, num_points, 3)
        sparse_xyz: tensor (batch_size, num_points2, 3), if None, then assume sparse_xyz = dense_xyz
        dense_feats: tensor (batch_size, num_points, num_dims)
        nei_inds: tensor (batch_size, num_points2, K)
        """
        B, N, _ = dense_xyz.shape
        if sparse_xyz is not None:
            _, M, _ = sparse_xyz.shape
        else:
            M = N
        _, _, K = nei_inds.shape

        # First downscaling mlp
        feats_x = self.unary1(dense_feats)

        gathered_xyz = index_points(dense_xyz, nei_inds)
        # localized_xyz = gathered_xyz - sparse_xyz.view(B, M, 1, D) #[B, M, K,
        # D]
        if sparse_xyz is not None:
            localized_xyz = gathered_xyz - sparse_xyz.unsqueeze(dim=2)
        else:
            localized_xyz = gathered_xyz - dense_xyz.unsqueeze(dim=2)
        gathered_norm = index_points(dense_xyz_norm, nei_inds)

        feat_pe = self.pe_convs(localized_xyz)  # [B, M, K, D]

        if self.cfg.USE_VI is True:
            if vi_features is None:
                if sparse_xyz is not None:
                    weightNetInput = VI_coordinate_transform(
                        localized_xyz, gathered_norm, sparse_xyz_norm, K)
                else:
                    weightNetInput = VI_coordinate_transform(
                        localized_xyz, gathered_norm, dense_xyz_norm, K)
            else:
                weightNetInput = vi_features
        else:
            weightNetInput = localized_xyz

        # If not using CUDA kernel, then we need to sparse gather the features
        # here
        if not self.cfg.USE_CUDA_KERNEL:
            gathered_feat = index_points(feats_x, nei_inds)  # [B, M, K, in_ch]
            new_feat = torch.cat([gathered_feat, feat_pe], dim=-1)

        weights = self.weightnet(weightNetInput)

        if self.cfg.USE_CUDA_KERNEL:
            feats_x = feats_x.contiguous()
            nei_inds = nei_inds.contiguous()
            weights = weights.contiguous()
            feat_pe = feat_pe.contiguous()
            new_feat = PConv.forward(feats_x, nei_inds, weights, feat_pe)
        else:
            new_feat = torch.matmul(
                input=new_feat.permute(
                    0, 1, 3, 2), other=weights).view(
                B, M, -1)

        new_feat = self.linear(new_feat)
        new_feat = F.relu(new_feat, inplace=True)

        # Dropout
        new_feat = self.dropout(new_feat)

        # Second upscaling mlp
        new_feat = self.unary2(new_feat)
        if sparse_xyz is not None:
            sparse_feats = torch.max(
                index_points(
                    dense_feats,
                    nei_inds),
                dim=2)[0]
        else:
            sparse_feats = dense_feats

        shortcut = self.unary_shortcut(sparse_feats)

        new_feat = self.leaky_relu(self.drop_path(new_feat) + shortcut)

        return new_feat, weightNetInput


class PointConv(nn.Module):
    '''
    This layer implements VI_PointConv and PointConv (set USE_VI = false) WITHOUT the bottleneck layer and without position encoding as features
    We use this only for the first layer, where input dimensionality is 3 and there is no point to use bottleneck
    Parameters:
        in_channel: Number of input channels
        out_channel: Number of output channels
        weightnet: Number of input/output channels for weightnet
        USE_VI: If not specified, then cfg.USE_VI is adopted, otherwise this overwrites cfg.USE_VI
    Input:
        dense_xyz: tensor (batch_size, num_points, 3). The coordinates of the points before subsampling (if it 
                   is a "strided" convolution wihch simultaneously subsamples the point cloud)
        dense_feats: tensor (batch_size, num_points, num_dims). The features of the points before subsampling.
        nei_inds: tensor (batch_size, num_points2, K). The neighborhood indices of the K nearest neighbors of 
                  each point (after subsampling). The indices should index into dense_xyz and dense_feats,
                  as during subsampling features at new coordinates are aggregated from the points before subsampling
        dense_xyz_norm: tensor (batch_size, num_points, 3). The surface normals of the points before subsampling
        sparse_xyz: tensor (batch_size, num_points2, 3). The coordinates of the points after subsampling (if there 
                    is no subsampling, just input None for this and the next)
        sparse_xyz_norm: tensor (batch_size, num_points2, 3). The surface normals of the points after subsampling
        vi_features: tensor (batch_size, num_points2, 12). VI features only needs to be computed once per stage. 
                     If it has been computed in a previous layer, it can be saved and directly inputted here.
        Note: batch_size is usually 1 since we are using the packed representation packing multiple point clouds into 
              one. However this dimension needs to be there for pyTorch to work properly.
    Output:
        new_feat: output features
        weightNetInput: the input to weightNet, which are relative coordinates or viewpoint-invariance aware transforms of it
    '''
    def __init__(
            self,
            in_channel,
            out_channel,
            cfg,
            weightnet=[9, 16],
            USE_VI=None):
        super(PointConv, self).__init__()
        self.cfg = cfg
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.USE_VI = cfg.USE_VI
        if USE_VI is not None:
            self.USE_VI = USE_VI
        last_ch = in_channel
        if cfg.USE_PE:
            if self.USE_VI:
                last_ch = in_channel + 12
            else:
                last_ch = in_channel + 3
        else:
            last_ch = in_channel
        self.weightnet = WeightNet(weightnet[0], weightnet[1], efficient=True)
        if cfg.BATCH_NORM:
            self.linear = Linear_BN(
                last_ch * weightnet[-1], out_channel, bn_ver='1d')
        else:
            self.linear = nn.Linear(last_ch * weightnet[-1], out_channel)

        self.dropout = nn.Dropout(
            p=cfg.dropout_rate) if cfg.dropout_rate > 0. else nn.Identity()

    def forward(
            self,
            dense_xyz,
            dense_feats,
            nei_inds,
            dense_xyz_norm=None,
            sparse_xyz=None,
            sparse_xyz_norm=None):
        """
        dense_xyz: tensor (batch_size, num_points, 3)
        sparse_xyz: tensor (batch_size, num_points2, 3)
        dense_feats: tensor (batch_size, num_points, num_dims)
        nei_inds: tensor (batch_size, num_points2, K)
        dense_xyz_norm: normals of the dense xyz, tensor (batch_size, num_points, 3)
        sparse_xyz_norm: normals of the sparse xyz, tensor (batch_size, num_points2, 3)
        norms are required if USE_VI is true
        """
        B, N, _ = dense_xyz.shape
        if sparse_xyz is not None:
            _, M, _ = sparse_xyz.shape
        else:
            M = N
        _, _, K = nei_inds.shape

        # nei_inds = nei_inds.clone().detach()
        # nei_inds_mask = (nei_inds != -1).float()
        # nn_idx_divider = nei_inds_mask.sum(dim = -1)
        # nn_idx_divider[nn_idx_divider == 0] = 1
        # nei_inds[nei_inds == -1] = 0

        gathered_xyz = index_points(dense_xyz, nei_inds)
        # localized_xyz = gathered_xyz - sparse_xyz.view(B, M, 1, D) #[B, M, K,
        # D]
        if sparse_xyz is not None:
            localized_xyz = gathered_xyz - sparse_xyz.unsqueeze(dim=2)
        else:
            localized_xyz = gathered_xyz - dense_xyz.unsqueeze(dim=2)

        if self.USE_VI is True:
            gathered_norm = index_points(dense_xyz_norm, nei_inds)
            if sparse_xyz is not None:
                weightNetInput = VI_coordinate_transform(
                    localized_xyz, gathered_norm, sparse_xyz_norm, K)
            else:
                weightNetInput = VI_coordinate_transform(
                    localized_xyz, gathered_norm, dense_xyz_norm, K)
        else:
            weightNetInput = localized_xyz

        gathered_feat = index_points(dense_feats, nei_inds)  # [B, M, K, in_ch]
        if self.cfg.USE_PE:
            gathered_feat = torch.cat([gathered_feat, weightNetInput], dim=-1)

        weights = self.weightnet(weightNetInput)

        # localized_xyz = localized_xyz.permute(0, 3, 2, 1)
        # weights = self.weightnet(localized_xyz)*nei_inds_mask.permute(0,2,1).unsqueeze(dim=1)
        new_feat = torch.matmul(
            input=gathered_feat.permute(
                0, 1, 3, 2), other=weights).view(
            B, M, -1)
        # new_feat = new_feat/nn_idx_divider.unsqueeze(dim=-1)
        new_feat = F.relu(self.linear(new_feat), inplace=True)

        # Dropout
        new_feat = self.dropout(new_feat)

        return new_feat, weightNetInput


class PointConvTransposePE(nn.Module):
    '''
    PointConvTranspose (upsampling) layer
    one needs to input dense_xyz (high resolution point coordinates after upsampling) and sparse_xyz (low-resolution) 
    and this layer would put features to the points at dense_xyz
    Parameters:
        in_channel: Number of input channels
        out_channel: Number of output channels
        weightnet: Number of input/output channels for weightnet
        mlp2: MLP after the PointConvTranspose

    Input:
        sparse_xyz: tensor (batch_size, num_points, 3). The coordinates of the points before upsampling
        sparse_feats: tensor (batch_size, num_points, num_dims). The features of the points before upsampling.
        nei_inds: tensor (batch_size, num_points2, K). The neighborhood indices of the K nearest neighbors of each 
                  point after upsampling. The indices should index into sparse_xyz and sparse_feats,
                  as during upsampling features at new coordinates are aggregated from the points before upsampling
        sparse_xyz_norm: tensor (batch_size, num_points, 3). The surface normals of the points before upsampling
        dense_xyz: tensor (batch_size, num_points2, 3). The coordinates of the points after upsampling (if there is no 
                   upsampling, just input None for this and the next)
        dense_xyz_norm: tensor (batch_size, num_points2, 3). The surface normals of the points after upsampling
        dense_feats: shortcut dense features
        vi_features: tensor (batch_size, num_points2, 12). VI features only needs to be computed once per stage. If it 
                     has been computed in a previous layer, it can be saved and directly inputted here.
        Note: batch_size is usually 1 since we are using the packed representation packing multiple point clouds into 
              one. However this dimension needs to be there for pyTorch to work properly.
    Output:
        new_feat: output features
        weightNetInput: the input to weightNet, which are relative coordinates or viewpoint-invariance aware transforms of it
    '''
    def __init__(
            self,
            in_channel,
            out_channel,
            cfg,
            weightnet=[9, 16],
            mlp2=None):
        super(PointConvTransposePE, self).__init__()
        self.cfg = cfg
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.drop_path = DropPath(
            cfg.drop_path_rate) if cfg.drop_path_rate > 0. else nn.Identity()
# This part can save a bit of memory, maybe with some performance drop or maybe no drop at all
#        self.unary1 = UnaryBlock(
#            in_channel,
#            out_channel,
#            use_bn=True,
#            bn_momentum=0.1)

        # positonal encoder
        self.pe_convs = nn.ModuleList()
        if cfg.USE_PE:
            self.pe_convs = WeightNet(
                3, min(out_channel // 4, 32), hidden_unit=[out_channel // 4], efficient=True)
            last_ch = min(out_channel // 4, 32)
        else:
            self.pe_convs = nn.ModuleList()
            last_ch = 0

        self.weightnet = WeightNet(weightnet[0], weightnet[1], efficient=True)
        if cfg.BATCH_NORM:
            # self.linear = Linear_BN(
            #                 (last_ch + out_channel) * weightnet[-1], out_channel, bn_ver='1d')
            self.linear = Linear_BN((last_ch + in_channel) * weightnet[-1], out_channel, bn_ver='1d')
        else:
            self.linear = nn.Linear((last_ch + in_channel) * weightnet[-1], out_channel, bn_ver='1d')
#            self.linear = nn.Linear(
#                (last_ch + out_channel) * weightnet[-1], out_channel)

        self.dropout = nn.Dropout(
            p=cfg.dropout_rate) if cfg.dropout_rate > 0. else nn.Identity()

        self.mlp2_convs = nn.ModuleList()
        self.mlp2_bns = nn.ModuleList()
        if mlp2 is not None:
            for i in range(1, len(mlp2)):
                if cfg.BATCH_NORM:
                    self.mlp2_convs.append(
                        Linear_BN(mlp2[i - 1], mlp2[i], bn_ver='1d'))
                else:
                    self.mlp2_convs.append(nn.Linear(mlp2[i - 1], mlp2[i]))

    def forward(
            self,
            sparse_xyz,
            sparse_feats,
            nei_inds,
            sparse_xyz_norm,
            dense_xyz,
            dense_xyz_norm,
            dense_feats=None,
            vi_features=None):
        """
        dense_xyz: tensor (batch_size, num_points, 3)
        sparse_xyz: tensor (batch_size, num_points2, 3)
        dense_feats: tensor (batch_size, num_points, num_dims)
        nei_inds: tensor (batch_size, num_points2, K)
        """
        B, _, _ = sparse_xyz.shape
        _, M, _ = dense_xyz.shape
        _, _, K = nei_inds.shape

        gathered_xyz = index_points(sparse_xyz, nei_inds)
        localized_xyz = gathered_xyz - dense_xyz.unsqueeze(dim=2)
        gathered_norm = index_points(sparse_xyz_norm, nei_inds)

        if self.cfg.USE_PE:
            feat_pe = self.pe_convs(localized_xyz)
        if self.cfg.USE_VI is True:
            if vi_features is None:
                weightNetInput = VI_coordinate_transform(
                    localized_xyz, gathered_norm, dense_xyz_norm, K)
            else:
                weightNetInput = vi_features
        else:
            weightNetInput = localized_xyz

        # feats_x = self.unary1(sparse_feats)
        feats_x = sparse_feats

        if not self.cfg.USE_CUDA_KERNEL:
            gathered_feat = index_points(feats_x, nei_inds)  # [B, M, K, in_ch]
            if self.cfg.USE_PE:
                gathered_feat = torch.cat([gathered_feat, feat_pe], dim=-1)

        weights = self.weightnet(weightNetInput)

        if self.cfg.USE_CUDA_KERNEL:
            feats_x = feats_x.contiguous()
            nei_inds = nei_inds.contiguous()
            weights = weights.contiguous()
            if self.cfg.USE_PE:
                feat_pe = feat_pe.contiguous()
                new_feat = PConv.forward(feats_x, nei_inds, weights, feat_pe)
            else:
                new_feat = PConv.forward(feats_x, nei_inds, weights)
        else:
            new_feat = torch.matmul(
                input=gathered_feat.permute(
                    0, 1, 3, 2), other=weights).view(
                B, M, -1)

        new_feat = F.relu(self.linear(new_feat), inplace=True)

        if dense_feats is not None:
            new_feat = new_feat + dense_feats

        # Dropout
        new_feat = self.dropout(new_feat)

        for conv in self.mlp2_convs:
            new_feat = F.relu(conv(new_feat), inplace=True)

        return new_feat, weightNetInput
