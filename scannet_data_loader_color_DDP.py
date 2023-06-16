#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022-2023 Apple Inc. All Rights Reserved.
#
# Data Loader for ScanNet

import random
import torch
import numpy as np
import glob
from torch.utils.data import Dataset
import transforms as t
from util.voxelize import voxelize

from datasetCommon import subsample_and_knn, compute_weight, collect_fn


class ScanNetDataset(Dataset):
    def __init__(self, cfg, dataset="training", rotate_deg=0.0, rotate_aug=False, flip_aug=False, scale_aug=False, transform_aug=False, color_aug=False, crop=False, shuffle_index=False, mix3D=False, voxelize_mode='random'):

        self.data = []
        self.cfg = cfg
        self.set = dataset

        self.rotate_aug = rotate_aug
        self.flip_aug = flip_aug
        self.scale_aug = scale_aug
        self.transform = transform_aug
        self.trans_std = [0.02, 0.02, 0.02]
        self.color_aug = color_aug
        self.crop = crop
        self.shuffle_index = shuffle_index
        self.mix3D = mix3D
        self.rotate_deg = rotate_deg
        self.voxelize_mode = voxelize_mode

        if 'noisy_points' in self.cfg and self.cfg.noisy_points:
            self.noisy_points = t.NoisyPoints(self.cfg.noise_level, self.cfg.noise_pct, self.cfg.ignore_label)
            self.add_noise = True
        else:
            self.add_noise = False

#        if self.color_aug:
            '''
            color_transform = [t.ChromaticAutoContrast(),
                               t.ChromaticTranslation(0.05),
                               t.ChromaticJitter(0.05)]
            '''
        # Random drop color is so important just let it be there now 
        if self.color_aug:
            color_transform = [t.RandomDropColor()]
            self.color_transform = t.Compose(color_transform)

        if self.set == "training":
            data_files = glob.glob(self.cfg.train_data_path)
        elif self.set == "validation":
            data_files = glob.glob(self.cfg.val_data_path)
        elif self.set == "trainval":
            data_files_train = glob.glob(self.cfg.train_data_path)
            data_files_val = glob.glob(self.cfg.val_data_path)
            data_files = data_files_train + data_files_val
        else:  # 'test'
            data_files = glob.glob(self.cfg.test_data_path)

        for x in torch.utils.data.DataLoader(
                data_files,
                collate_fn=lambda x: torch.load(x[0]), num_workers=cfg.NUM_WORKERS):
            self.data.append(x)

        print('%s examples: %d' % (self.set, len(self.data)))

        if self.cfg.USE_WEIGHT:
            weights = compute_weight(self.data)
        else:
            weights = [1.0] * 20
        print("label weights", weights)
        self.cfg.weights = weights

    def __len__(self):
        """
        Return the length of data here
        """
        return len(self.data)

    def _augment_data(self, coord, color, norm, label):
        #########################
        # random data augmentation 
        #########################

        # random augmentation by rotation
        if self.rotate_aug:
            # rotate_rad = np.deg2rad(np.random.random() * 360) - np.pi
            rotate_rad = np.deg2rad(torch.rand(1).numpy()[0] * 360) - np.pi
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            coord[:, :2] = np.dot(coord[:, :2], j)
            norm[:, :2] = np.dot(norm[:, :2], j)
            # print(rotate_rad)

        # random augmentation by flip x, y or x+y
        if self.flip_aug:
            flip_type = torch.randint(4, (1,)).numpy()[0]  # np.random.choice(4, 1)
            # print(flip_type)
            if flip_type == 1:
                coord[:, 0] = -coord[:, 0]
                norm[:, 0] = -norm[:, 0]
            elif flip_type == 2:
                coord[:, 1] = -coord[:, 1]
                norm[:, 1] = -norm[:, 1]
            elif flip_type == 3:
                coord[:, :2] = -coord[:, :2]
                norm[:, :2] = -norm[:, :2]

        if self.scale_aug:
            noise_scale = torch.rand(1).numpy()[0] * 0.4 + 0.8  # np.random.uniform(0.8, 1.2)
            # print(noise_scale)
            coord[:, 0] = noise_scale * coord[:, 0]
            coord[:, 1] = noise_scale * coord[:, 1]

        if self.transform:
            # noise_translate = np.array([np.random.normal(0, self.trans_std[0], 1),
            #                             np.random.normal(0, self.trans_std[1], 1),
            #                             np.random.normal(0, self.trans_std[2], 1)]).T
            # noise_translate = np.array([torch.randn(1).numpy()[0] * self.trans_std[0],
            #                             torch.randn(1).numpy()[0] * self.trans_std[1],
            #                             torch.randn(1).numpy()[0] * self.trans_std[2]]).T
            num_points = coord.shape[0]
            noise_translate = torch.randn(num_points, 3).numpy()
            noise_translate[:, 0] *= self.trans_std[0]
            noise_translate[:, 1] *= self.trans_std[1]
            noise_translate[:, 2] *= self.trans_std[2]
            # print("before range: ", coord.min(0), coord.max(0))
            coord[:, 0:3] += noise_translate
            # print("after range: ", coord.min(0), coord.max(0))
            # print(noise_translate)

        if self.color_aug:
            #            color = (color + 1) * 127.5
            #            color *= 255.
            coord, color, label, norm = self.color_transform(coord, color, label, norm)
#            color = color / 127.5 - 1
#            color /= 255.

        # crop half of the scene
        if self.crop:
            points = coord - coord.mean(0)
            if torch.rand(1).numpy()[0] < 0.5:
                inds = np.all([points[:, 0] >= 0.0], axis=0)
            else:
                inds = np.all([points[:, 0] < 0.0], axis=0)

            coord, color, norm, label = (
                coord[~inds],
                color[~inds],
                norm[~inds],
                label[~inds]
            )

        return coord, color, norm, label

    def get_raw_coord(self, indx):
        return self.data[indx][0]

    def get_scene_name(self, indx):
        return self.data[indx][3]

    def __getitem__(self, indx):
        coord, features, label, _ = self.data[indx]

        color, norm = features[:, :3], features[:, 3:]

        # move z to 0+
        z_min = coord[:, 2].min()
        coord[:, 2] -= z_min 

        # Specific rotation
        if self.rotate_deg != 0.:
            rotate_rad = np.deg2rad(self.rotate_deg * 360) - np.pi
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            coord[:, :2] = np.dot(coord[:, :2], j)
            norm[:, :2] = np.dot(norm[:, :2], j)

        coord, color, norm, label = self._augment_data(coord, color, norm, label)

        # Add noise to examine robustness
        if self.add_noise:
            coords, color, norm, label = self.noisy_points(coord, color, norm, label)


        # Mix3D augmentation from 3DV 2021 paper
        if self.mix3D and random.random() < 0.8:
            mix_idx = np.random.randint(len(self.data))
            coords2, features2, labels2, _ = self.data[mix_idx]

            color2, norm2 = features2[:, :3], features2[:, 3:]
            z_min = coords2[:, 2].min()
            coords2[:, 2] -= z_min
            coords2, color2, norm2, labels2 = self._augment_data(coords2, color2, norm2, labels2)
            coord = np.concatenate((coord, coords2), axis=0)
            color = np.concatenate((color, color2), axis=0)
            norm = np.concatenate((norm, norm2), axis=0)
            label = np.concatenate((label, labels2), axis=0)

        # input normalize
        coord_min = np.min(coord, 0)
        coord -= coord_min

        # voxelize coords
        uniq_idx = voxelize(coord, self.cfg.grid_size[0], mode=self.voxelize_mode)
        if self.voxelize_mode != 'multiple':
            all_data = {}
            uniq_idx = voxelize(coord, self.cfg.grid_size[0], mode=self.voxelize_mode)
            coord, color, norm, label = coord[uniq_idx], color[uniq_idx], norm[uniq_idx], label[uniq_idx]
        else:
            all_data = []
            # For testing set, output multiple sets of data for the purpose of testing on all points before voxelization
            for crop_idx in uniq_idx:
                data = {}
                coord_part, norm_part = coord[crop_idx], norm[crop_idx]
                point_list, nei_forward_list, nei_propagate_list, nei_self_list, norm_list \
                    = subsample_and_knn(coord_part, norm_part, grid_size=self.cfg.grid_size, K_self=self.cfg.K_self,
                                        K_forward=self.cfg.K_forward, K_propagate=self.cfg.K_propagate)
                data['point_list'] = point_list
                data['nei_forward_list'] = nei_forward_list
                data['nei_propagate_list'] = nei_propagate_list
                data['nei_self_list'] = nei_self_list
                data['surface_normal_list'] = norm_list
                data['feature_list'] = [color[crop_idx].astype(np.float32)]
                data['label_list'] = [label[crop_idx].astype(np.int32)]
                data['crop_idx'] = crop_idx
                all_data.append(data)
            return all_data

        # subsample points during training time
        if (self.set == "training" or self.set == "trainval") and label.shape[0] > self.cfg.MAX_POINTS_NUM:
            init_idx = torch.randint(label.shape[0], (1,)).numpy()[0]
            crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:self.cfg.MAX_POINTS_NUM]
            coord, color, norm, label = coord[crop_idx], color[crop_idx], norm[crop_idx], label[crop_idx]

        # shuffle_index
        if self.shuffle_index:
            shuf_idx = torch.randperm(coord.shape[0]).numpy()
            coord, color, norm, label = coord[shuf_idx], color[shuf_idx], norm[shuf_idx], label[shuf_idx]


        # print("number of points: ", coord.shape[0])

        point_list, nei_forward_list, nei_propagate_list, nei_self_list, norm_list = \
            subsample_and_knn(coord, norm, grid_size=self.cfg.grid_size, K_self=self.cfg.K_self,
                              K_forward=self.cfg.K_forward, K_propagate=self.cfg.K_propagate)
        all_data['point_list'] = point_list
        all_data['nei_forward_list'] = nei_forward_list
        all_data['nei_propagate_list'] = nei_propagate_list
        all_data['nei_self_list'] = nei_self_list
        all_data['surface_normal_list'] = norm_list
        all_data['feature_list'] = [color.astype(np.float32)]
        all_data['label_list'] = [label.astype(np.int32)]

        return all_data


def getdataLoadersDDP(cfg):
    '''
    Get ScanNet data loaders for the training and validation set. If cfg.DDP is set then a DistributedSampler is used, otherwise, a RandomSampler is used
    Input: 
        cfg: config dictionary
    Output:
        train_loader: data loader for the training set
        val_loader: data loader for the validation set
    '''
    # Initialize datasets
    if cfg.DDP:
        train_loader, val_loader, _, _ = getdataLoaders(cfg, torch.utils.data.distributed.DistributedSampler)
    else:
        train_loader, val_loader, _, _ = getdataLoaders(cfg, torch.utils.data.RandomSampler)
    return train_loader, val_loader


def getdataLoaders(cfg, sampler):
    '''
    Get ScanNet data loaders with the correct augmentations
    Output:
        train_loader: data loader for the training set
        val_loader: data loader for the validation set
        train_dataset: a ScanNetDataset object for the training set
        validation_dataset: a ScanNetDataset object for the validation set
    '''
    # Initialize datasets
    training_dataset = ScanNetDataset(cfg, dataset="training", 
                                      rotate_aug=cfg.rotate_aug,
                                      flip_aug=cfg.flip_aug,
                                      scale_aug=cfg.scale_aug,
                                      transform_aug=cfg.transform_aug,
                                      color_aug=cfg.color_aug,
                                      crop=cfg.crop, 
                                      shuffle_index=cfg.shuffle_index,
                                      mix3D=cfg.mix3D)
    # No data augmentation for validation
    validation_dataset = ScanNetDataset(cfg, dataset="validation")

    training_sampler = sampler(training_dataset)

    validation_sampler = sampler(validation_dataset)

    train_data_loader = torch.utils.data.DataLoader(training_dataset,
                                                    batch_size=cfg.BATCH_SIZE, 
                                                    collate_fn=collect_fn, 
                                                    num_workers=cfg.NUM_WORKERS, 
                                                    pin_memory=True,
                                                    sampler=training_sampler,
                                                    drop_last=True)

    val_data_loader = torch.utils.data.DataLoader(validation_dataset,
                                                  batch_size=cfg.BATCH_SIZE, 
                                                  collate_fn=collect_fn, 
                                                  num_workers=cfg.NUM_WORKERS, 
                                                  sampler=validation_sampler,
                                                  pin_memory=True)

    return train_data_loader, val_data_loader, training_dataset, validation_dataset
