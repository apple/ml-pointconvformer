#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022-2023 Apple Inc. All Rights Reserved.
#


# Testing code with voting, can be used to benchmark on the testing set
# Usage: python test_ScanNet_voting.py --config config_file --pretrain_path model --split split
#        where config_file is the training config file (e.g. configFLPCF_10cm.yaml, model is the .pth file
#        stored from training, and split is the split you want to test on (e.g. 'validation'))

import os
import time
import argparse
import datetime
import yaml
import numpy as np
from easydict import EasyDict as edict

import torch
import torch.nn.functional as F
import torch.utils.data

import open3d as o3d

from util.common_util import AverageMeter, intersectionAndUnion, replace_batchnorm, to_device, init_seeds
from util.logger import get_logger

from model_architecture import PointConvFormer_Segmentation as Main_Model
from train_ScanNet_DDP_WarmUP import get_default_configs
from datasetCommon import collect_fn
from scannet_data_loader_color_DDP import ScanNetDataset


def collect_fn_test(data_list):
    '''
    Crop the points from a data list into several different ones if there are too many points.
    Besides the normal collect function, additionally return idx_data for the different point crops.
     (check the 'multiple' mode of the voxelization)
    '''
    all_features = []
    all_pointclouds = []
    all_edges_self = []
    all_edges_forward = []
    all_edges_propagate = []
    all_target = []
    all_norms = []
    idx_data = []
    for data in data_list:
        the_start = 0
        while the_start < len(data):
            count = 0
            the_end = len(data)
            crop_idx = np.zeros(0)
            for i, crop in enumerate(data[the_start:]):
                count += len(crop['crop_idx'])
                if count > args.MAX_POINTS_NUM:
                    the_end = the_start+i
                    break
                crop_idx = np.concatenate((crop_idx, crop['crop_idx']))
            features, pointclouds, edges_self, edges_forward, edges_propagate, target, norms = collect_fn(data[the_start:the_end])
            all_features.append(features)
            all_pointclouds.append(pointclouds)
            all_edges_self.append(edges_self)
            all_edges_forward.append(edges_forward)
            all_edges_propagate.append(edges_propagate)
            all_target.append(target)
            all_norms.append(norms)
            idx_data.append(crop_idx)
            the_start = the_end

    return all_features, all_pointclouds, all_edges_self, all_edges_forward, \
        all_edges_propagate, all_target, all_norms, idx_data


def get_parser():
    '''
    Get the arguments of the call.
    '''
    parser = argparse.ArgumentParser(
        description='PointConvFormer for Semantic Segmentation')
    parser.add_argument(
        '--config',
        default='./configFLPCF_10cm.yaml',
        type=str,
        help='config file')
    parser.add_argument(
        '--pretrain_path',
        default='./',
        type=str,
        help='the path of pretrain weights')
    parser.add_argument(
        '--vote_num',
        default=1,
        type=int,
        help='number of vote')
    parser.add_argument(
        '--split',
        default='validation',
        type=str,
        help='the path of pretrain weights')
    parser.add_argument(
        '--init_deg',
        default=0.,
        type=float,
        help='inital degree [0 ~ 1]')

    args = parser.parse_args()
    assert args.config is not None
    cfg = edict(yaml.safe_load(open(args.config, 'r')))
    cfg = get_default_configs(cfg)
    cfg.pretrain_path = args.pretrain_path
    cfg.config = args.config
    cfg.vote_num = args.vote_num
    cfg.split = args.split
    cfg.init_deg = args.init_deg
    return cfg


def main_vote():
    global args, logger
    args = get_parser()

    file_dir = os.path.join(args.eval_path, 'TIME_%s_SemSeg-' %
                            (args.model_name) +
                            str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
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
    model = Main_Model(args)

    logger.info(model)

    init_seeds(args.manual_seed)

    if os.path.isfile(args.pretrain_path):
        logger.info("=> loading checkpoint '{}'".format(args.pretrain_path))
        checkpoint = torch.load(args.pretrain_path)
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict, strict=True)
        logger.info(
            "=> loaded checkpoint '{}' (epoch {})".format(
                args.pretrain_path,
                checkpoint['epoch']))
        args.epoch = checkpoint['epoch']
    else:
        raise RuntimeError(
            "=> no checkpoint found at '{}'".format(
                args.pretrain_path))

    logger.info(
        '>>>>>>>>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>>>>>>>>>>>')
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    colormap = np.array(create_color_palette40()) / 255.0
    mapper = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                      12, 14, 16, 24, 28, 33, 34, 36, 39])

    model.eval()
    replace_batchnorm(model)
    model = model.to('cuda')

    num_vote = args.vote_num
    rotate_deg_list = np.arange(num_vote) / num_vote + args.init_deg
    logger.info("rotate_degs: {}".format(rotate_deg_list))

    torch.cuda.empty_cache()

    output_dict = {}

    args.BATCH_SIZE = 1

    pcd = o3d.geometry.PointCloud()
    time_list = []

    for vote_id in range(num_vote):

        print(
            "vote_id : ",
            vote_id,
            "rotate_deg : ",
            rotate_deg_list[vote_id] *
            360)
        rotate_deg = rotate_deg_list[vote_id]
        # Rotate differently for each vote
        val_dataset = ScanNetDataset(args, dataset=args.split, rotate_deg=rotate_deg, voxelize_mode='multiple')
        dataset_loader = torch.utils.data.DataLoader(val_dataset,
                                                     batch_size=args.BATCH_SIZE, 
                                                     collate_fn=collect_fn_test, 
                                                     num_workers=args.NUM_WORKERS, 
                                                     sampler=torch.utils.data.SequentialSampler(val_dataset),
                                                     pin_memory=True)

        rotate_deg = int(rotate_deg_list[vote_id] * 360)
        cur_prob_dir = os.path.join(prob_dir, str(rotate_deg))
        if not os.path.exists(cur_prob_dir):
            os.makedirs(cur_prob_dir)

        for idx, all_data in enumerate(dataset_loader):
            scene_name = val_dataset.get_scene_name(idx)
            features, pointclouds, edges_self, edges_forward, edges_propagate, label, norms, idx_data = all_data
            features, pointclouds, edges_self, edges_forward, edges_propagate, label, norms = \
                to_device(features), to_device(pointclouds), \
                to_device(edges_self), to_device(edges_forward), \
                to_device(edges_propagate), to_device(label), to_device(norms)
            all_size = int(max([item.max() for item in idx_data]) + 1)
            pred = torch.zeros((all_size, args.num_classes)).cuda()
            all_labels = torch.zeros((all_size), dtype=torch.long).cuda()
            for i in range(len(features)):
                with torch.no_grad():
                    torch.cuda.synchronize()
                    st = time.time()
                    pred_part = model(features[i], pointclouds[i], edges_self[i], edges_forward[i], edges_propagate[i], norms[i])

                    torch.cuda.synchronize()
                    et = time.time()
                    time_list.append(et - st)
                    pred_part = pred_part.contiguous().view(-1, args.num_classes)
                    pred_part = F.softmax(pred_part, dim=-1)

                    torch.cuda.empty_cache()
                    pred[idx_data[i], :] += pred_part
                    all_labels[idx_data[i]] = label[i]
                    logger.info('Test: {}/{}, {}, {}, {}'.format(idx + 1, 
                                len(dataset_loader), 
                                i + 1, idx_data[i].shape[0], et - st))

            pred = pred / (pred.sum(-1)[:, None] + 1e-8)
            # pred = pred.max(1)[1].data.cpu().numpy()
            raw_coord = val_dataset.get_raw_coord(idx)

            np.save(os.path.join(cur_prob_dir, scene_name + '.npy'),
                    {'pred': pred, 'target': label, 'xyz': raw_coord})

            if scene_name in output_dict:
                output_dict[scene_name]['pred'] += pred
            else:
                cur_dict = {
                    'pred': pred,
                    'target': all_labels,
                    'xyz': raw_coord
                }
                output_dict[scene_name] = cur_dict
            torch.cuda.empty_cache()

    pcd = o3d.geometry.PointCloud()
    for scene_name, output_data in output_dict.items():
        print('scene_name : ', scene_name)
        target = output_data['target'].cpu().numpy().astype(np.int32)
        pred = output_data['pred'].cpu().numpy()
        xyz = output_data['xyz']

        output = pred.argmax(axis=1)

        intersection, union, target = intersectionAndUnion(
            output, target, args.num_classes, args.ignore_label)
        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target)

        output = mapper[output.astype(np.int32)]
        pcd.points = o3d.utility.Vector3dVector(xyz)
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
    logger.info(
        'Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    for i in range(args.num_classes):
        logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i,
                    iou_class[i], accuracy_class[i]))
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
