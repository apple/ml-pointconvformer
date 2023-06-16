#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022-2023 Apple Inc. All Rights Reserved.
#

import torch
import numpy as np
from sklearn.neighbors import KDTree
import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
try:
    import cpp_wrappers.cpp_neighbors.radius_neighbors as cpp_neighbors
except BaseException:
    print('Failed to import cpp_neighbors, nanoflann kNN is not loaded. Only sklearn kNN is available')


def grid_subsampling(
        points,
        features=None,
        labels=None,
        sampleDl=0.1,
        verbose=0):
    """
    This function comes from https://github.com/HuguesTHOMAS/KPConv
    Copyright Hugues Thomas 2018
    CPP wrapper for a grid subsampling (method = barycenter for points and features
    Input:
        points: (N, 3) matrix of input points
        features: optional (N, d) matrix of features (floating number)
        labels: optional (N,) matrix of integer labels
        sampleDl: parameter defining the size of grid voxels
        verbose: 1 to display
        subsampled points, with features and/or labels depending of the input
    Output: subsampled points
    """

    # method = "voxelcenters" # "barycenters" "voxelcenters"
    method = "barycenters"

    if (features is None) and (labels is None):
        return cpp_subsampling.compute(
            points,
            sampleDl=sampleDl,
            verbose=verbose,
            method=method)
    elif (labels is None):
        return cpp_subsampling.compute(
            points,
            features=features,
            sampleDl=sampleDl,
            verbose=verbose,
            method=method)
    elif (features is None):
        return cpp_subsampling.compute(
            points,
            classes=labels,
            sampleDl=sampleDl,
            verbose=verbose,
            method=method)
    else:
        return cpp_subsampling.compute(
            points,
            features=features,
            classes=labels,
            sampleDl=sampleDl,
            verbose=verbose,
            method=method)


def compute_weight(train_data, num_class=20):
    """
    Compute the class weights for ScanNet classes based on square root of inverse proportional weighting
    Input:
        training data: (points, features, labels, norms), only labels are used
    Output:
        a list of weights for each class
    """
    weights = np.array([0.0 for i in range(num_class)])

    num_rooms = len(train_data)
    for i in range(num_rooms):
        _, _, labels, _ = train_data[i]
        # rm invalid labels
        labels = labels[labels >= 0]
        for j in range(num_class):
            weights[j] += np.sum(labels == j)

    ratio = weights / float(sum(weights))
    ce_label_weight = 1 / (np.power(ratio, 1 / 2))
    return list(ce_label_weight)


def compute_knn(ref_points, query_points, K, dilated_rate=1, method='sklearn'):
    """
    Compute KNN
    Input:
        ref_points: reference points (MxD)
        query_points: query points (NxD)
        K: the amount of neighbors for each point
        dilated_rate: If set to larger than 1, then select more neighbors and then choose from them
        (Engelmann et al. Dilated Point Convolutions: On the Receptive Field Size of Point Convolutions on 3D Point Clouds. ICRA 2020)
        method: Choose between two approaches: Scikit-Learn ('sklearn') or nanoflann ('nanoflann'). In general nanoflann should be faster, but sklearn is more stable
    Output:
        neighbors_idx: for each query point, its K nearest neighbors among the reference points (N x K)
    """
    num_ref_points = ref_points.shape[0]

    if num_ref_points < K or num_ref_points < dilated_rate * K:
        num_query_points = query_points.shape[0]
        inds = np.random.choice(
            num_ref_points, (num_query_points, K)).astype(
            np.int32)

        return inds
    if method == 'sklearn':
        kdt = KDTree(ref_points)
        neighbors_idx = kdt.query(
            query_points,
            k=K * dilated_rate,
            return_distance=False)
    elif method == 'nanoflann':
        neighbors_idx = batch_neighbors(
            query_points, ref_points, [
                query_points.shape[0]], [num_ref_points], K * dilated_rate)
    else:
        raise Exception('compute_knn: unsupported knn algorithm')
    if dilated_rate > 1:
        neighbors_idx = np.array(
            neighbors_idx[:, ::dilated_rate], dtype=np.int32)

    return neighbors_idx


def crop(points, x_min, y_min, z_min, x_max, y_max, z_max):
    """
    Crop all points within a 3D bounding box defined by (x_min, y_min, z_min) and (x_max, y_max, z_max)
    Input:
        points: an array of points (nx3)
        x_min, y_min, z_min, x_max, y_max, z_max: bounding box extremal coordinates
    Output:
        inds: indices of the points that are within the bounding box defined by (x_min, y_min, z_min) and (x_max, y_max, z_max)
    """
    if x_max <= x_min or y_max <= y_min or z_max <= z_min:
        raise ValueError(
            "We should have x_min < x_max and y_min < y_max and z_min < z_max. But we got"
            " (x_min = {x_min}, y_min = {y_min}, z_min = {z_min},"
            " x_max = {x_max}, y_max = {y_max}, z_max = {z_max})".format(
                x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, z_min=z_min, z_max=z_max, ))
    inds = np.all(
        [
            (points[:, 0] >= x_min),
            (points[:, 0] < x_max),
            (points[:, 1] >= y_min),
            (points[:, 1] < y_max),
            (points[:, 2] >= z_min),
            (points[:, 2] < z_max),
        ],
        axis=0,
    )
    return inds


def tensorizeList(nplist, is_index=False):
    """
    Make all numpy arrays inside a list into torch tensors
    """
    ret_list = []
    for npitem in nplist:
        if is_index:
            if npitem is None:
                ret_list.append(None)
            else:
                ret_list.append(
                    torch.from_numpy(
                        npitem).long().unsqueeze(0))
        else:
            ret_list.append(torch.from_numpy(npitem).float().unsqueeze(0))

    return ret_list


def tensorize(
        features,
        pointclouds,
        edges_self,
        edges_forward,
        edges_propagate,
        target,
        norms):
    """
    Convert numpy arrays from inside lists into torch tensors for all input data
    """
    pointclouds = tensorizeList(pointclouds)
    norms = tensorizeList(norms)
    edges_self = tensorizeList(edges_self, True)
    edges_forward = tensorizeList(edges_forward, True)
    edges_propagate = tensorizeList(edges_propagate, True)

    target = torch.from_numpy(target).long().unsqueeze(0)
    features = torch.from_numpy(features).float().unsqueeze(0)

    return features, pointclouds, edges_self, edges_forward, edges_propagate, target, norms


def listToBatch(
        features,
        pointclouds,
        edges_self,
        edges_forward,
        edges_propagate,
        target,
        norms):
    """
    ListToBatch transforms a batch of multiple clouds into one point cloud so that we do not have to pad them to the same length
    The way this works is that all point clouds are concatenated one after another e.g., if you have point cloud 1 which is [5154,3], point cloud 2 which is [4749, 3]
    then it creates a point cloud as if it has batch size 1, which is a tensor of shape [1, 5154+4749, 3]
    It also modifies the edges (indices of k-nearest-neighbors) so that they point to the correct points
    For example, for point cloud 2, we add 5154 to all its neighbor indices so that they
    link to the points in point cloud 2 in this combined tensor
    Input: List versions of all the input
    Output: Batched versions of all the input
    """
    # import ipdb; ipdb.set_trace()
    num_sample = len(pointclouds)

    # process sample 0
    featureBatch = features[0][0]
    pointcloudsBatch = pointclouds[0]
    pointcloudsNormsBatch = norms[0]
    if target:
        targetBatch = target[0][0]
    else:
        targetBatch = np.array(0)

    edgesSelfBatch = edges_self[0]
    edgesForwardBatch = edges_forward[0]
    edgesPropagateBatch = edges_propagate[0]

    points_stored = [val.shape[0] for val in pointcloudsBatch]

    for i in range(1, num_sample):
        if target:
            targetBatch = np.concatenate([targetBatch, target[i][0]], 0)
        featureBatch = np.concatenate([featureBatch, features[i][0]], 0)

        for j in range(len(edges_forward[i])):
            tempMask = edges_forward[i][j] == -1
            edges_forwardAdd = edges_forward[i][j] + points_stored[j]
            edges_forwardAdd[tempMask] = -1
            edgesForwardBatch[j] = np.concatenate([edgesForwardBatch[j],
                                                   edges_forwardAdd], 0)

            tempMask2 = edges_propagate[i][j] == -1
            edges_propagateAdd = edges_propagate[i][j] + points_stored[j + 1]
            edges_propagateAdd[tempMask2] = -1
            edgesPropagateBatch[j] = np.concatenate([edgesPropagateBatch[j],
                                                     edges_propagateAdd], 0)

        for j in range(len(pointclouds[i])):
            tempMask3 = edges_self[i][j] == -1
            edges_selfAdd = edges_self[i][j] + points_stored[j]
            edges_selfAdd[tempMask3] = -1
            edgesSelfBatch[j] = np.concatenate([edgesSelfBatch[j],
                                                edges_selfAdd], 0)

            pointcloudsBatch[j] = np.concatenate(
                [pointcloudsBatch[j], pointclouds[i][j]], 0)
            pointcloudsNormsBatch[j] = np.concatenate(
                [pointcloudsNormsBatch[j], norms[i][j]], 0)

            points_stored[j] += pointclouds[i][j].shape[0]

    return featureBatch, pointcloudsBatch, edgesSelfBatch, edgesForwardBatch, edgesPropagateBatch, \
        targetBatch, pointcloudsNormsBatch


def prepare(
        features,
        pointclouds,
        edges_self,
        edges_forward,
        edges_propagate,
        target,
        norms):
    """
    Prepare data coming from data loader (lists of numpy arrays) into torch tensors ready to send to training
    """

    features_out, pointclouds_out, edges_self_out, edges_forward_out, edges_propagate_out, target_out, norms_out = [], [], [], [], [], [], []

    features_out, pointclouds_out, edges_self_out, edges_forward_out, edges_propagate_out, target_out, norms_out = \
        listToBatch(features, pointclouds, edges_self, edges_forward, edges_propagate, target, norms)

    features_out, pointclouds_out, edges_self_out, edges_forward_out, edges_propagate_out, target_out, norms_out = \
        tensorize(features_out, pointclouds_out, edges_self_out, edges_forward_out, edges_propagate_out, target_out, norms_out)

    return features_out, pointclouds_out, edges_self_out, edges_forward_out, edges_propagate_out, target_out, norms_out


def collect_fn(data_list):
    """
    collect data from the data dictionary and outputs pytorch tensors
    """
    features = []
    pointclouds = []
    target = []
    norms = []
    edges_forward = []
    edges_propagate = []
    edges_self = []
    for i, data in enumerate(data_list):
        features.append(data['feature_list'])
        pointclouds.append(data['point_list'])
        if 'label_list' in data.keys():
            target.append(data['label_list'])
        norms.append(data['surface_normal_list'])

        edges_forward.append(data['nei_forward_list'])
        edges_propagate.append(data['nei_propagate_list'])
        edges_self.append(data['nei_self_list'])

    features, pointclouds, edges_self, edges_forward, edges_propagate, target, norms = \
        prepare(features, pointclouds, edges_self, edges_forward, edges_propagate, target, norms)

    return features, pointclouds, edges_self, edges_forward, edges_propagate, target, norms


def subsample_and_knn(
        coord,
        norm,
        grid_size=[0.1],
        K_self=16,
        K_forward=16,
        K_propagate=16):
    """
    Perform grid subsampling and compute kNN at each subsampling level
    Input:
        coord: N x 3 coordinates
        norm: N x 3 surface normals
        grid_size: all the downsampling levels (in cm) you want to use, e.g. [0.05, 0.1, 0.2, 0.4, 0.8]
        K_self: number of neighbors within each downsampling level
        K_forward: number of neighbors from one downsampling level to the next one (with less points), used in the downsampling PointConvs in the encoder
        K_propagate: number of neighbors from one downsampling level to the previous one (with more points), used in the upsampling PointConvs in the decoder
    Outputs:
        point_list: list of length len(grid_size)
        nei_forward_list: downsampling kNN neighbors (K_forward neighbors for each point)
        nei_propagate_list: upsampling kNN neighbors (K_propagate neighbors for each point)
        nei_self_list: kNN neighbors between the same layer
        norm_list: list of surface normals averaged within each voxel at each grid_size
    """
    point_list, norm_list = [], []
    nei_forward_list, nei_propagate_list, nei_self_list = [], [], []

    for j, grid_s in enumerate(grid_size):
        if j == 0:
            sub_point, sub_norm = coord.astype(
                np.float32), norm.astype(
                np.float32)

            point_list.append(sub_point)
            norm_list.append(sub_norm)
            # compute edges
            nself = compute_knn(sub_point, sub_point, K_self[j])
            nei_self_list.append(nself)

        else:
            sub_point, sub_norm = grid_subsampling(
                points=point_list[-1], features=norm_list[-1], sampleDl=grid_s)

            if sub_point.shape[0] <= K_self[j]:
                sub_point, sub_norm = point_list[-1], norm_list[-1]

            # compute edges, nforward is for downsampling, npropagate is for upsampling,
            # nself is for normal PointConv layers (between the same set of
            # points)
            nforward = compute_knn(point_list[-1], sub_point, K_forward[j])
            npropagate = compute_knn(sub_point, point_list[-1], K_propagate[j])
            nself = compute_knn(sub_point, sub_point, K_self[j])
            # point_list is a list with len(grid_size) length, each item is a numpy array
            # of num_points x dimensionality

            point_list.append(sub_point)
            norm_list.append(sub_norm)
            nei_forward_list.append(nforward)
            nei_propagate_list.append(npropagate)
            nei_self_list.append(nself)

    return point_list, nei_forward_list, nei_propagate_list, nei_self_list, norm_list


def getdataLoader(cfg, dataset, loader_set, sampler):
    """
    Dataset should generate items as a dictionary with:
        color_list: colors
        point_list: point xyz coordinates
        surface_normal_list: surface normals
        nei_forward_list: neighbors from upper level to lower level
        nei_propagate_list: neighbors from lower level to upper level
        nei_self_list: neighbors at the same level
        label_list: optional labels, if no labels is given
    These can be prepared by the subsample_and_knn function which takes the input of coordinates, color,
    surface normals and optional labels. Surface normals can be computed with Open3D, but could also be
    obtained e.g. from meshes etc.
    """
    this_dataset = dataset(cfg, dataset=loader_set)
    this_sampler = sampler(this_dataset)
    data_loader = torch.utils.data.DataLoader(
        this_dataset,
        batch_size=cfg.BATCH_SIZE,
        collate_fn=collect_fn,
        num_workers=cfg.NUM_WORKERS,
        sampler=this_sampler,
        pin_memory=True)
    return data_loader, this_dataset


def batch_neighbors(queries, supports, q_batches, s_batches, K):
    """
    Computes neighbors for a batch of queries and supports
    :param queries: (N1, 3) the query points
    :param supports: (N2, 3) the support points
    :param q_batches: (B) the list of lengths of batch elements in queries
    :param s_batches: (B)the list of lengths of batch elements in supports
    :param K: long
    :return: neighbors indices
    """

    return cpp_neighbors.batch_kquery(
        queries, supports, q_batches, s_batches, K=int(K))
