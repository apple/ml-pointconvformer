import os
import time 
import random
import argparse
import numpy as np 
import shutil 
import yaml
from easydict import EasyDict as edict
import datetime 
import glob

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn 
import torch.nn.parallel 
import torch.optim 
import torch.utils.data 
import torch.nn.functional as F
from tensorboardX import SummaryWriter
try:
    import turibolt as bolt
except ModuleNotFoundError as err:
    # Error handling
    print(err)

from util.common_util import AverageMeter, intersectionAndUnionGPU
from util.logger import get_logger 
from util.lr import MultiStepWithWarmup, CosineAnnealingWarmupRestarts
from model_architecture import VI_PointConvGuidedPENewAfterDropout as VI_PointConv
import scannet_data_loader_color_DDP as scannet_data_loader
from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter

def get_default_configs(cfg):
    cfg.num_level = len(cfg.grid_size)
    if cfg.feat_dim is None:
        cfg.feat_dim  = [cfg.base_dim * (i + 1) for i in range(cfg.num_level + 1)]
    return cfg

def to_device(input, device='cuda'):

    if isinstance(input, list) and len(input) > 0:
        for idx in range(len(input)):
            input[idx] = input[idx].to(device)
    
    if isinstance(input, torch.Tensor):
        input = input.to(device)

    return input 

def get_parser():
    parser = argparse.ArgumentParser('ScanNet PointConvFormer')
    parser.add_argument('--local_rank', default=-1, type=int, help='local_rank')
    parser.add_argument('--config', default='./configWenxuanPCFDDPL5WarmUP.yaml', type=str, help='config file')

    args = parser.parse_args()
    assert args.config is not None
    cfg = edict(yaml.safe_load(open(args.config, 'r')))
    cfg = get_default_configs(cfg)
    cfg.local_rank = args.local_rank
    cfg.config = args.config
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

def main_process():
    return not args.DDP or (args.DDP and args.rank % args.num_gpus == 0)

def main():
    args = get_parser()
    args.use_cuda = True

    validation_dataset = scannet_data_loader.ScanNetDataset(args,set="validation")
    validation_sampler = torch.utils.data.RandomSampler(validation_dataset)
    val_data_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=1, collate_fn=scannet_data_loader.collect_fn,num_workers=0, pin_memory=True, sampler=validation_sampler,drop_last=True)

    # get model 
    model = VI_PointConv(args)
    model = model.to('cuda')
    # TODO: load saved weights!

    ############################
    # start training  #
    ############################

    for i, data in enumerate(val_data_loader):
        features, pointclouds, edges_self, edges_forward, edges_propagate, target, norms = data
        if args.use_cuda:
            features, pointclouds, edges_self, edges_forward, edges_propagate, target, norms = \
            to_device(features), to_device(pointclouds), \
            to_device(edges_self), to_device(edges_forward), \
            to_device(edges_propagate), to_device(target), to_device(norms)
        with torch.no_grad():
            pred = model(features, pointclouds, edges_self, edges_forward, edges_propagate, norms)
        ret_layers = {'pcf_backbone.pointconv.1.unary1':'layer1_out','pcf_backbone.pointconv.1.weightnet':'weightnet_out','pcf_backbone.pointconv.1.guidance_weight':'guidance'}
        mid_getter = MidGetter(model, return_layers = ret_layers, keep_output=False)
        mid_output,_ = mid_getter(features,pointclouds,edges_self,edges_forward,edges_propagate, norms)
        layer1_out = mid_output['layer1_out']
        weightnet_out = mid_output['weightnet_out']
        guidance = mid_output['guidance']
        linear_weight = model.pcf_backbone.pointconv[1].linear.weight
        mlp_weight = model.pcf_backbone.pointconv[1].unary2.mlp.weight
        to_save = {'layer1_out':mid_output['layer1_out'],'weightnet_out':mid_output['weightnet_out'],'guidance':mid_output['guidance'],'linear_weight':model.pcf_backbone.pointconv[1].linear.weight,'mlp_weight':model.pcf_backbone.pointconv[1].unary2.mlp.weight,'neighbor_inds':edges_forward[1]}
        torch.save(to_save,'test_files.pth')
        break

if __name__ == '__main__':
    main()
