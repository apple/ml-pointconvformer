import os
import time
import random
import numpy as np
import logging
import pickle
import argparse
import collections
import datetime
import yaml
from easydict import EasyDict as edict
from tqdm import tqdm
import glob
import copy

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from torch.utils.data import Dataset


import open3d as o3d

from util.common_util import AverageMeter, intersectionAndUnion, check_makedirs, replace_batchnorm, to_device, init_seeds, intersectionAndUnionGPU

from model_architecture import PointConvFormer_Segmentation as VI_PointConv
from util.voxelize import voxelize
from util.logger import get_logger
from scannet_data_loader_color_DDP import ScanNetDataset
from datasetCommon import getdataLoader
from train_ScanNet_DDP_WarmUP import get_default_configs

def get_parser():
    parser = argparse.ArgumentParser(description='PointConvFormer for Semantic Segmentation')
    parser.add_argument('--config', default='./configFLPCF_10cm.yaml', type=str, help='config file')
    parser.add_argument('--pretrain_path', default='./', type=str, help='the path of pretrain weights')
    parser.add_argument('--split', default='validation', type=str, help='the dataset split to be tested on')


    args = parser.parse_args()
    assert args.config is not None
    cfg = edict(yaml.safe_load(open(args.config, 'r')))
    cfg = get_default_configs(cfg)
    cfg.pretrain_path = args.pretrain_path
    cfg.config = args.config
    cfg.split = args.split
    return cfg


def collect_fns(data_list):
    return data_list


MAX_NUM_POINTS = 550000


def main_vote():
    global args, logger
    args = get_parser()

    file_dir = os.path.join(args.eval_path, 'TIME_%s_SemSeg-'%(args.model_name) + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
    args.file_dir = file_dir
    ply_dir = os.path.join(args.file_dir, 'ply')
    txt_dir = os.path.join(args.file_dir, 'txt')
    prob_dir = os.path.join(args.file_dir, 'prob')

    if not os.path.exists(args.eval_path):
        os.makedirs(args.eval_path)

    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    if not os.path.exists(ply_dir):
        os.makedirs(ply_dir)

    if not os.path.exists(txt_dir):
        os.makedirs(txt_dir)

    if not os.path.exists(prob_dir):
        os.makedirs(prob_dir)

    logger = get_logger(file_dir)
    logger.info(args)
    assert args.num_classes > 1
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))

    # get model
    model = VI_PointConv(args)

    logger.info(model)

    init_seeds(args.manual_seed)

    if os.path.isfile(args.pretrain_path):
        logger.info("=> loading checkpoint '{}'".format(args.pretrain_path))
        checkpoint = torch.load(args.pretrain_path)
        state_dict = checkpoint['state_dict']
        new_state_dict = collections.OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=True)
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.pretrain_path, checkpoint['epoch']))
        args.epoch = checkpoint['epoch']
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.pretrain_path))

    logger.info('>>>>>>>>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>>>>>>>>>>>')
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    colormap = np.array(create_color_palette40()) / 255.0
    mapper = np.array([1,2,3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39])

    model.eval()
    replace_batchnorm(model)
    model = model.to('cuda')
    
    # file_dir = os.path.join(args.experiment_dir, '%s_SemSeg-'%(args.model_name) + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
    # args.file_dir = file_dir
    # check_makedirs(file_dir)
    # import ipdb; ipdb.set_trace()

    torch.cuda.empty_cache()

    output_dict = {}
    args.BATCH_SIZE = 1
    val_data_loader, val_dataset =  getdataLoader(args, ScanNetDataset, args.split, torch.utils.data.SequentialSampler)

#    dataset = ScanNetDataset(args, split=args.split)

    # import ipdb; ipdb.set_trace()
    pcd = o3d.geometry.PointCloud()
    time_list = []

    for idx, data in enumerate(val_data_loader):
        scene_name = val_dataset.get_scene_name(idx)
        features, pointclouds, edges_self, edges_forward, edges_propagate, label, norms = data
        features, pointclouds, edges_self, edges_forward, edges_propagate, label, norms = \
            to_device(features), to_device(pointclouds), \
            to_device(edges_self), to_device(edges_forward), \
            to_device(edges_propagate), to_device(label), to_device(norms)
           
        with torch.no_grad():
            torch.cuda.synchronize()
            st = time.time()
            pred = model(features, pointclouds, edges_self, edges_forward, edges_propagate, norms)
            torch.cuda.synchronize()
            et = time.time()
            time_list.append(et - st)
            pred = pred.contiguous().view(-1, args.num_classes)
            pred = F.softmax(pred, dim = -1)
        torch.cuda.empty_cache()
        logger.info('Test: {}/{}, {}'.format(idx + 1, len(val_data_loader), et - st))



        np.save(os.path.join(prob_dir, scene_name+'.npy'), {'pred': pred, 'target': label, 'xyz': pointclouds[0]})

        cur_dict = {
                    'pred': pred,
                    'target': label,
                    'xyz': pointclouds[0]
                   }
        output_dict[scene_name] = cur_dict


    # import ipdb; ipdb.set_trace()
    pcd = o3d.geometry.PointCloud()
    for scene_name, output_data in output_dict.items():
        print('scene_name : ', scene_name)
        labels = output_data['target'].view(-1, 1)[:, 0]

        pred = output_data['pred']
        xyz = output_data['xyz'].cpu().numpy()

        output = pred.max(1)[1]
#        intersection, union, target = intersectionAndUnionGPU(output, labels, args.num_classes)
#        print(intersection)
        labels = labels.cpu().numpy().astype(np.int32)
        output = output.cpu().numpy()

        #import ipdb; ipdb.set_trace()
        intersection, union, target = intersectionAndUnion(output, labels, args.num_classes, args.ignore_label)
#        print(intersection2)

        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        output = mapper[output.astype(np.int32)]
        pcd.points = o3d.utility.Vector3dVector(xyz[0])
        pcd.colors = o3d.utility.Vector3dVector(colormap[output])
        o3d.io.write_point_cloud(str(ply_dir) + '/' + scene_name, pcd)

        fp = open(str(txt_dir) + '/' + scene_name.replace("_vh_clean_2.ply", ".txt"), "w")
        for l in output:
            fp.write(str(int(l)) + '\n')
        fp.close()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    for i in range(args.num_classes):
        logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

    logger.info('Average running time per frame: ', np.mean(time_list))


def create_color_palette40():
    return [
       (0, 0, 0),
       (174, 199, 232),		# wall
       (152, 223, 138),		# floor
       (31, 119, 180), 		# cabinet
       (255, 187, 120),		# bed
       (188, 189, 34), 		# chair
       (140, 86, 75),  		# sofa
       (255, 152, 150),		# table
       (214, 39, 40),  		# door
       (197, 176, 213),		# window
       (148, 103, 189),		# bookshelf
       (196, 156, 148),		# picture
       (23, 190, 207), 		# counter
       (178, 76, 76),
       (247, 182, 210),		# desk
       (66, 188, 102),
       (219, 219, 141),		# curtain
       (140, 57, 197),
       (202, 185, 52),
       (51, 176, 203),
       (200, 54, 131),
       (92, 193, 61),
       (78, 71, 183),
       (172, 114, 82),
       (255, 127, 14), 		# refrigerator
       (91, 163, 138),
       (153, 98, 156),
       (140, 153, 101),
       (158, 218, 229),		# shower curtain
       (100, 125, 154),
       (178, 127, 135),
       (120, 185, 128),
       (146, 111, 194),
       (44, 160, 44),  		# toilet
       (112, 128, 144),		# sink
       (96, 207, 209),
       (227, 119, 194),		# bathtub
       (213, 92, 176),
       (94, 106, 211),
       (82, 84, 163),  		# otherfurn
       (100, 85, 144)
    ]


if __name__ == '__main__':
    main_vote()



