#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022-2023 Apple Inc. All Rights Reserved.
#

import numpy as np
import torch


def fnv_hash_vec(arr):
    """
    FNV64-1A
    """
    assert arr.ndim == 2
    # Floor first for negative coordinates
    arr = arr.copy()
    arr = arr.astype(np.uint64, copy=False)
    hashed_arr = np.uint64(14695981039346656037) * np.ones(arr.shape[0], dtype=np.uint64)
    for j in range(arr.shape[1]):
        hashed_arr *= np.uint64(1099511628211)
        hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
    return hashed_arr


def ravel_hash_vec(arr):
    """
    Ravel the coordinates after subtracting the min coordinates.
    """
    assert arr.ndim == 2
    arr = arr.copy()
    arr -= arr.min(0)
    arr = arr.astype(np.uint64, copy=False)
    arr_max = arr.max(0).astype(np.uint64) + 1

    keys = np.zeros(arr.shape[0], dtype=np.uint64)
    # Fortran style indexing
    for j in range(arr.shape[1] - 1):
        keys += arr[:, j]
        keys *= arr_max[j + 1]
    keys += arr[:, -1]
    return keys


def voxelize(coord, voxel_size=0.05, hash_type='fnv', mode='random'):
    '''
    Voxelization of the input coordinates
    Parameters:
        coord: input coordinates (N x D)
        voxel_size: Size of the voxels
        hash_type: Type of the hashing function, can be chosen from 'ravel' and 'fnv'
        mode: 'random', 'deterministic' or 'multiple' mode. In training mode one selects a random point within the voxel as the representation of the voxel.
              In deterministic model right now one always uses the first point. Usually random mode is preferred for training. In 'multiple' mode, we will return
              multiple sets of indices, so that each point will be covered in at least one of these sets
    Returns:
        idx_unique: the indices of the points so that there is at most one point for each voxel
    '''
    discrete_coord = np.floor(coord / np.array(voxel_size))
    if hash_type == 'ravel':
        key = ravel_hash_vec(discrete_coord)
    else:
        key = fnv_hash_vec(discrete_coord)

    idx_sort = np.argsort(key)
    key_sort = key[idx_sort]
    _, count = np.unique(key_sort, return_counts=True)
    if mode == 'deterministic':
        # idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + torch.randint(count.max(), (count.size,)).numpy() % count
        idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + np.zeros((count.size,), dtype=np.int32)
        idx_unique = idx_sort[idx_select]
        return idx_unique
    elif mode == 'multiple':  # mode is 'multiple'
        idx_data = []
        for i in range(count.max()):
            idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
            idx_part = idx_sort[idx_select]
            idx_data.append(idx_part)
        return idx_data
    else:  # mode == 'random'
        # idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + np.random.randint(0, count.max(), count.size) % count
        idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + torch.randint(count.max(), (count.size,)).numpy() % count
        idx_unique = idx_sort[idx_select]
        return idx_unique
