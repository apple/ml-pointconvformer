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

from sklearn.neighbors import KDTree
import open3d as o3d

from util.common_util import AverageMeter, intersectionAndUnion, check_makedirs, replace_batchnorm

from model_architecture import VI_PointConvGuidedPENewAfterDropout as VI_PointConv
from util.voxelize import voxelize
from util.logger import get_logger
from scannet_data_loader_color_DDP import compute_knn, grid_subsampling, tensorlizeList

from torch.profiler import profile, record_function, ProfilerActivity


def get_default_configs(cfg):
    cfg.num_level = len(cfg.grid_size)
    if cfg.feat_dim is None:
        cfg.feat_dim  = [cfg.base_dim * (i + 1) for i in range(cfg.num_level + 1)]
    return cfg

def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def to_device(input, device='cuda'):

    if isinstance(input, list) and len(input) > 0:
        for idx in range(len(input)):
            input[idx] = input[idx].to(device)

    if isinstance(input, torch.Tensor):
        input = input.to(device)

    return input


def get_parser():
    parser = argparse.ArgumentParser(description='PointConvFormer for Semantic Segmentation')
    parser.add_argument('--config', default='./configFLPCF_10cm.yaml', type=str, help='config file')
    parser.add_argument('--pretrain_path', default='./', type=str, help='the path of pretrain weights')
    parser.add_argument('--vote_num', default=1, type=int, help='number of vote')
    parser.add_argument('--split', default='validation', type=str, help='the path of pretrain weights')
    parser.add_argument('--init_deg', default=0., type=float, help='inital degree [0 ~ 1]')


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


def input_normalize(coord):
    coord_min = np.min(coord, 0)
    coord -= coord_min
    return coord

def preprocess_pointcloud(coord, color, norm, args):
    # input normalize
    coord_min = np.min(coord, 0)
    coord -= coord_min

    # print("number of points: ", coord.shape[0])
    point_list, color_list, norm_list = [], [], []
    nei_forward_list, nei_propagate_list, nei_self_list = [], [], []

    for j, grid_s in enumerate(args.grid_size):
        if j == 0:
            # sub_point, sub_norm_color, sub_labels = \
            #     grid_subsampling(points=coord.astype(np.float32), features=np.concatenate((norm, color), axis=1).astype(np.float32), \
            #                      labels=label.astype(np.int32), sampleDl=grid_s)
            sub_point, sub_color, sub_norm = coord.astype(np.float32), color.astype(np.float32), norm.astype(np.float32)

            point_list.append(sub_point)
            color_list.append(sub_color)
            norm_list.append(sub_norm)

            # compute edges
            nself = compute_knn(sub_point, sub_point, args.K_self[j])
            nei_self_list.append(nself)

        else:
            sub_point, sub_norm = \
                grid_subsampling(points=point_list[-1], features=norm_list[-1], sampleDl=grid_s)

            # if sub_point.shape[0] <= args.K_self[j]:
            #     sub_point, sub_norm = point_list[-1], norm_list[-1]

            # compute edges
            nforward = compute_knn(point_list[-1], sub_point, args.K_forward[j])
            npropagate = compute_knn(sub_point, point_list[-1], args.K_propagate[j])
            nself = compute_knn(sub_point, sub_point, args.K_self[j])

            point_list.append(sub_point)
            norm_list.append(sub_norm)
            nei_forward_list.append(nforward)
            nei_propagate_list.append(npropagate)
            nei_self_list.append(nself)

    return color_list, point_list, nei_forward_list, nei_propagate_list, nei_self_list, norm_list

def tensorlize(features, pointclouds, edges_self, edges_forward, edges_propagate, norms):
    pointclouds = tensorlizeList(pointclouds)
    norms = tensorlizeList(norms)
    edges_self = tensorlizeList(edges_self, True)
    edges_forward = tensorlizeList(edges_forward, True)
    edges_propagate = tensorlizeList(edges_propagate, True)

    features = torch.from_numpy(features).float().unsqueeze(0)

    return features, pointclouds, edges_self, edges_forward, edges_propagate, norms

def listToBatch(features, pointclouds, edges_self, edges_forward, edges_propagate, norms):
    # import ipdb; ipdb.set_trace()
    num_sample = len(pointclouds)

    #process sample 0
    featureBatch = features[0][0]
    pointcloudsBatch = pointclouds[0]
    pointcloudsNormsBatch = norms[0]

    edgesSelfBatch = edges_self[0]
    edgesForwardBatch = edges_forward[0]
    edgesPropagateBatch = edges_propagate[0]

    points_stored = [val.shape[0] for val in pointcloudsBatch]

    for i in range(1, num_sample):
        featureBatch = np.concatenate([featureBatch, features[i][0]], 0)

        for j in range(len(edges_forward[i])):
            tempMask = edges_forward[i][j] == -1
            edges_forwardAdd = edges_forward[i][j] + points_stored[j]
            edges_forwardAdd[tempMask] = -1
            edgesForwardBatch[j] = np.concatenate([edgesForwardBatch[j], \
                                   edges_forwardAdd], 0)

            tempMask2 = edges_propagate[i][j] == -1
            edges_propagateAdd = edges_propagate[i][j] + points_stored[j + 1]
            edges_propagateAdd[tempMask2] = -1
            edgesPropagateBatch[j] = np.concatenate([edgesPropagateBatch[j], \
                                   edges_propagateAdd], 0)

        for j in range(len(pointclouds[i])):
            tempMask3 = edges_self[i][j] == -1
            edges_selfAdd = edges_self[i][j] + points_stored[j]
            edges_selfAdd[tempMask3] = -1
            edgesSelfBatch[j] = np.concatenate([edgesSelfBatch[j], \
                                    edges_selfAdd], 0)

            pointcloudsBatch[j] = np.concatenate([pointcloudsBatch[j], pointclouds[i][j]], 0)
            pointcloudsNormsBatch[j] = np.concatenate([pointcloudsNormsBatch[j], norms[i][j]], 0)

            points_stored[j] += pointclouds[i][j].shape[0]

    return featureBatch, pointcloudsBatch, edgesSelfBatch, edgesForwardBatch, edgesPropagateBatch, \
           pointcloudsNormsBatch


def prepare(features, pointclouds, edges_self, edges_forward, edges_propagate, norms):

    features_out, pointclouds_out, edges_self_out, edges_forward_out, edges_propagate_out, norms_out = [], [], [], [], [], []

    features_out, pointclouds_out, edges_self_out, edges_forward_out, edges_propagate_out, norms_out = \
        listToBatch(features, pointclouds, edges_self, edges_forward, edges_propagate, norms)

    features_out, pointclouds_out, edges_self_out, edges_forward_out, edges_propagate_out, norms_out = \
        tensorlize(features_out, pointclouds_out, edges_self_out, edges_forward_out, edges_propagate_out, norms_out)

    return features_out, pointclouds_out, edges_self_out, edges_forward_out, edges_propagate_out, norms_out


def collect_fn(data_list):
    # import ipdb; ipdb.set_trace()
    features = []
    pointclouds = []
    norms = []
    edges_forward = []
    edges_propagate = []
    edges_self = []
    for idx, data in enumerate(data_list):

        feature_list, point_list, nei_self_list, nei_forward_list, nei_propagate_list, normal_list = data
        features.append(feature_list)
        pointclouds.append(point_list)
        norms.append(normal_list)

        edges_forward.append(nei_forward_list)
        edges_propagate.append(nei_propagate_list)
        edges_self.append(nei_self_list)

    features, pointclouds, edges_self, edges_forward, edges_propagate, norms = \
            prepare(features, pointclouds, edges_self, edges_forward, edges_propagate, norms)

    return features, pointclouds, edges_self, edges_forward, edges_propagate, norms


def collect_fns(data_list):
    return data_list

class ScanNetDataset(Dataset):
    def __init__(self, cfg, split="training", rotate_deg = None):

        # rotate: [0, 360]

        self.data = []
        self.cfg = cfg
        self.set = split
        self.rotate_deg = rotate_deg

        if split == "training":
            data_files = glob.glob(self.cfg.train_data_path)
        elif split == "validation":
            data_files = glob.glob(self.cfg.val_data_path)
        elif split == "trainval":
            data_files = glob.glob(self.cfg.val_data_path) + glob.glob(self.cfg.train_data_path)
        else:
            data_files = glob.glob(self.cfg.test_data_path)

        for x in torch.utils.data.DataLoader(
                data_files,
                collate_fn=lambda x: torch.load(x[0]), num_workers=self.cfg.NUM_WORKERS):
            self.data.append(x)

        print('%s examples: %d'%(self.set, len(self.data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, indx):

        coord, features, label, scene_name = self.data[indx]
        raw_coord = copy.deepcopy(coord)

        color, norm = features[:, :3], features[:, 3:]
        # move z to 0+
        z_min = coord[:, 2].min()
        coord[:, 2] -= z_min

        # transform random
        rotate_rad = np.deg2rad(self.rotate_deg * 360) - np.pi
        c, s = np.cos(rotate_rad), np.sin(rotate_rad)
        j = np.matrix([[c, s], [-s, c]])
        coord[:, :2] = np.dot(coord[:, :2], j)
        norm[:, :2] = np.dot(norm[:, :2], j)

        idx_data = []
        idx_sort, count = voxelize(coord, self.cfg.grid_size[0], mode=1)
        for i in range(count.max()):
            idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
            idx_part = idx_sort[idx_select]
            idx_data.append(idx_part)

        '''
        idx_size = len(idx_data)
        idx_list, coord_list, color_list, norm_list, offset_list = [], [], [], [], []
        data_in_list = []

        for i in range(idx_size):
            idx_part = idx_data[i]
            coord_part, color_part, norm_part = coord[idx_part], color[idx_part], norm[idx_part]
            coord_part = input_normalize(coord_part)
            idx_list.append(idx_part), coord_list.append(coord_part), color_list.append(color_part), norm_list.append(norm_part)
            offset_list.append(idx_part.size)

            color_in, point_in, nei_forward_in, nei_propagate_in, nei_self_in, norm_in = \
                preprocess_pointcloud(coord_part, color_part, norm_part, args)

            data_in_list.append((color_in, point_in, nei_self_in, nei_forward_in, nei_propagate_in, norm_in))
        '''

        return idx_data, raw_coord, coord, color, norm, label, scene_name



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

    num_vote = args.vote_num
    rotate_deg_list = np.arange(num_vote) / num_vote + args.init_deg
    logger.info("rotate_degs: {}".format(rotate_deg_list) )


    torch.cuda.empty_cache()

    batch_size_test = 2

    output_dict = {}

    dataset = ScanNetDataset(args, split=args.split)

    # import ipdb; ipdb.set_trace()
    pcd = o3d.geometry.PointCloud()
    time_list = []

    for vote_id in range(num_vote):

        print("vote_id : ", vote_id, "rotate_deg : ", rotate_deg_list[vote_id] * 360)
        dataset.rotate_deg = rotate_deg_list[vote_id]
        dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=collect_fns, num_workers=0)

        rotate_deg = int(rotate_deg_list[vote_id] * 360)
        cur_prob_dir = os.path.join(prob_dir, str(rotate_deg))
        if not os.path.exists(cur_prob_dir):
            os.makedirs(cur_prob_dir)

        # for i, data in tqdm(enumerate(validation_dataset), total=len(validation_dataset), smoothing=0.9):
        for idx, data in enumerate(dataset_loader):

            idx_data, raw_coord, coord, color, norm, label, scene_name = data[0]
            idx_data = idx_data[:1]
            idx_size = len(idx_data)

            pred = torch.zeros((label.size, args.num_classes)).cuda()

            idx_list, coord_list, color_list, norm_list = [], [], [], []
            data_in_list = []
            # import ipdb; ipdb.set_trace()

            for i in range(idx_size):
                logger.info('{}/{}: {}/{}/{}, {}'.format(idx + 1, len(dataset_loader), i + 1, idx_size, idx_data[0].shape[0], scene_name))
                idx_part = idx_data[i]
                coord_part, color_part, norm_part = coord[idx_part], color[idx_part], norm[idx_part]
                if args.MAX_POINTS_NUM and coord_part.shape[0] > args.MAX_POINTS_NUM:
                    # import ipdb; ipdb.set_trace()

                    coord_p, idx_uni, cnt = np.random.rand(coord_part.shape[0]) * 1e-3, np.array([]), 0
                    while idx_uni.size != idx_part.shape[0]:
                        init_idx = np.argmin(coord_p)
                        dist = np.sum(np.power(coord_part - coord_part[init_idx], 2), 1)
                        idx_crop = np.argsort(dist)[:args.MAX_POINTS_NUM]
                        coord_sub, color_sub, norm_sub, idx_sub = coord_part[idx_crop], color_part[idx_crop], norm_part[idx_crop], idx_part[idx_crop]
                        dist = dist[idx_crop]
                        delta = np.square(1 - dist/np.max(dist))
                        coord_p[idx_crop] += delta
                        coord_sub = input_normalize(coord_sub)
                        idx_list.append(idx_sub), coord_list.append(coord_sub), color_list.append(color_sub)
                        norm_list.append(norm_sub)
                        idx_uni = np.unique(np.concatenate((idx_uni, idx_sub)))

                        color_in, point_in, nei_forward_in, nei_propagate_in, nei_self_in, norm_in = \
                            preprocess_pointcloud(coord_sub, color_sub, norm_sub, args)

                        data_in_list.append((color_in, point_in, nei_self_in, nei_forward_in, nei_propagate_in, norm_in))

                else:
                    coord_part = input_normalize(coord_part)
                    idx_list.append(idx_part), coord_list.append(coord_part), color_list.append(color_part), norm_list.append(norm_part)

                    color_in, point_in, nei_forward_in, nei_propagate_in, nei_self_in, norm_in = \
                        preprocess_pointcloud(coord_part, color_part, norm_part, args)

                    data_in_list.append((color_in, point_in, nei_self_in, nei_forward_in, nei_propagate_in, norm_in))

            # import ipdb; ipdb.set_trace()

            batch_size_test = 1 # int(np.floor(MAX_NUM_POINTS / args.MAX_POINTS_NUM))
            print("batch_size: ", batch_size_test)
            batch_num = int(np.ceil(len(idx_list) / batch_size_test))
            for i in range(batch_num):
                s_i, e_i = i * batch_size_test, min((i + 1) * batch_size_test, len(idx_list))
                idx_part = np.concatenate(idx_list[s_i:e_i])

                color_in, point_in, nei_self_in, nei_forward_in, nei_propagate_in, norm_in = \
                    collect_fn(data_in_list[s_i:e_i])

                color_in, point_in, nei_self_in, nei_forward_in, nei_propagate_in, norm_in = \
                    to_device(color_in), to_device(point_in), to_device(nei_self_in), \
                    to_device(nei_forward_in), to_device(nei_propagate_in), to_device(norm_in)

                with torch.no_grad():
                 with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA], record_shapes=True) as prof:
                    torch.cuda.synchronize()
                    st = time.time()
                    with record_function("model"):
                      pred_part = model(color_in, point_in, nei_self_in, nei_forward_in, nei_propagate_in, norm_in)
                    torch.cuda.synchronize()
                    et = time.time()
                    time_list.append(et - st)
                    pred_part = pred_part.contiguous().view(-1, args.num_classes)
                    pred_part = F.softmax(pred_part, dim = -1)
                 print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

                torch.cuda.empty_cache()
                pred[idx_part, :] += pred_part
                logger.info('Test: {}/{}, {}/{}, {}, {}'.format(idx + 1, len(dataset_loader), i + 1, batch_num, idx_part.shape[0], et - st))



            # import ipdb; ipdb.set_trace()

            pred = pred / (pred.sum(-1)[:, None] + 1e-8)
            # pred = pred.max(1)[1].data.cpu().numpy()

            np.save(os.path.join(cur_prob_dir, scene_name+'.npy'), {'pred': pred, 'target': label, 'xyz': raw_coord})

            if scene_name in output_dict:
                output_dict[scene_name]['pred'] += pred
            else:
                cur_dict = {
                    'pred': pred,
                    'target': label,
                    'xyz': raw_coord
                }
                output_dict[scene_name] = cur_dict


    # import ipdb; ipdb.set_trace()
    pcd = o3d.geometry.PointCloud()
    for scene_name, output_data in output_dict.items():
        print('scene_name : ', scene_name)
        target = output_data['target'].astype(np.int32)
        pred = output_data['pred'].cpu().numpy()
        xyz = output_data['xyz']

        output = pred.argmax(axis=1)

        # import ipdb; ipdb.set_trace()

        intersection, union, target = intersectionAndUnion(output, target, args.num_classes, args.ignore_label)
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

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



