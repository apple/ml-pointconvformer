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
import torch.multiprocessing as mp 
import torch.distributed as dist
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from tensorboardX import SummaryWriter
try:
    import turibolt as bolt
except ModuleNotFoundError as err:
    # Error handling
    print(err)

from util.common_util import AverageMeter, intersectionAndUnionGPU, to_device, init_seeds
from util.logger import get_logger 
from util.lr import MultiStepWithWarmup, CosineAnnealingWarmupRestarts
from loss import SmoothCrossEntropy
from model_architecture import PointConvFormer_Segmentation as VI_PointConv
from model_architecture import get_default_configs
import scannet_data_loader_color_DDP as scannet_data_loader

from torch.profiler import profile, record_function, ProfilerActivity

def get_default_training_cfgs(cfg):
    if 'label_smoothing' not in cfg.keys():
        cfg.label_smoothing = False
    if 'accum_iter' not in cfg.keys():
        cfg.accum_iter = 1
    return cfg

def get_parser():
    parser = argparse.ArgumentParser('ScanNet PointConvFormer')
    parser.add_argument('--local_rank', default=-1, type=int, help='local_rank')
    parser.add_argument('--config', default='./configWenxuanPCFDDPL5WarmUP.yaml', type=str, help='config file')

    args = parser.parse_args()
    assert args.config is not None
    cfg = edict(yaml.safe_load(open(args.config, 'r')))
    cfg = get_default_configs(cfg, cfg.num_level, cfg.base_dim)
    cfg = get_default_training_cfgs(cfg)
    cfg.local_rank = args.local_rank
    cfg.config = args.config
    return cfg


def main_process():
    return not args.DDP or (args.DDP and args.rank % args.num_gpus == 0)

def main():
    args = get_parser()

    if args.save_to_blobby:
        file_dir = os.path.join(bolt.ARTIFACT_DIR, '%s_SemSeg-'%(args.model_name) + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
        args.file_dir = file_dir
    else:
        file_dir = os.path.join(args.experiment_dir, '%s_SemSeg-'%(args.model_name) + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
        args.file_dir = file_dir

    code_dir = os.path.join(args.file_dir, 'code_log')

    if args.local_rank == 0:
        if not os.path.exists(args.experiment_dir):
            os.makedirs(args.experiment_dir)

        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        if not os.path.exists(code_dir):
            os.makedirs(code_dir)
        os.system('cp %s %s' % (args.config, code_dir))
        os.system('cp %s %s' % (os.path.basename(__file__), code_dir))
        os.system('cp %s %s' % ('scannet_data_loader_color_DDP.py', code_dir))
        os.system('cp %s %s' % ('model_architecture.py', code_dir))
        if args.DDP and not args.use_tensorboard:
            bolt.disable_grpc_channel_caching()

    print("ignore label: ", args.ignore_label)

    if 'manual_seed' in args.keys():
        init_seeds(args.manual_seed)

    args.distributed = args.DDP
    args.num_gpus = torch.cuda.device_count()

    main_worker(args)

def main_worker(config):
    global args, best_iou
    args, best_iou = config, 0 

    if torch.cuda.device_count() > 1:
        args.rank = int(os.environ["RANK"])
        local_rank = args.local_rank
        print("local_rank : ", local_rank)
        print("rank: ", args.rank)
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', world_size = args.num_gpus, rank=local_rank)
        dist.barrier()

        init_seeds(args.manual_seed + torch.distributed.get_rank())
    else:
        args.DDP = False
        init_seeds(args.manual_seed)
        local_rank = 0

    if main_process():
        global logger, writer
        if args.save_to_blobby:
            logger = get_logger(bolt.LOG_DIR)
        else:
            logger = get_logger(args.file_dir)
        logger.info(args)
        logger.info("=> Creating model ...")
        logger.info("Classes: {}".format(args.num_classes))
        writer = SummaryWriter(args.file_dir)


    train_data_loader, val_data_loader = scannet_data_loader.getdataLoadersDDP(args)

    # get model 
    model = VI_PointConv(args).to(local_rank)
    if main_process():
        logger.info(model)
        logger.info("#Model parameters: {}".format(sum([x.nelement() for x in model.parameters()])))

    # make model to DDP
    if args.DDP:
        if args.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        
    # get loss func
    if args.USE_WEIGHT:
        if main_process():
            logger.info("use weight!")
        if args.label_smoothing:
#            criterion = SmoothCrossEntropy(label_smoothing = args.label_smoothing,weight = args.weights, num_classes = args.num_classes, ignore_index = args.ignore_label).cuda()
            criterion = nn.CrossEntropyLoss(weight=torch.tensor(args.weights).float(), ignore_index=args.ignore_label, label_smoothing = args.label_smoothing).cuda()
        else:
            criterion = nn.CrossEntropyLoss(weight=torch.tensor(args.weights).float(), ignore_index=args.ignore_label).cuda()
    else:
        if args.label_smoothing:
#            criterion = SmoothCrossEntropy(label_smoothing = args.label_smoothing, num_classes = args.num_classes, ignore_index = args.ignore_label).cuda()
            criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label, label_smoothing = args.label_smoothing).cuda()
        else:
            criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda()

    val_criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda()

    # set optimizer 
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate)
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.adamw_decay)

        '''
        transformer_lr_scale = args.get("transformer_lr_scale", 0.1)
        param_dicts = [
            {"params": [p for n, p in model.named_parameters() if "guidance_weight" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model.named_parameters() if "guidance_weight" in n and p.requires_grad],
                "lr": args.learning_rate * transformer_lr_scale,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=args.learning_rate, weight_decay=args.adamw_decay)
        '''



    # optimizer.param_groups[0]['initial_lr'] = cfg.learning_rate
    init_epoch = 0

    # config lr scheduler

    iter_per_epoch = len(train_data_loader)

    if args.scheduler == "MultiStepWithWarmup":
        if args.milestones is not None:
            milestones = [v * iter_per_epoch for v in args.milestones]
        else:
            milestones = [int(args.total_epoches * 0.4) * iter_per_epoch, int(args.total_epoches* 0.6)*iter_per_epoch, int(args.total_epoches * 0.8)*iter_per_epoch]
        if main_process():
            logger.info("scheduler: MultiStepWarmup!!!")
            logger.info("milestones: {}".format(milestones))
        scheduler = MultiStepWithWarmup(optimizer, milestones=milestones, gamma=args.gamma, warmup=args.warmup, \
            warmup_iters=args.warmup_epochs*iter_per_epoch, warmup_ratio=args.warmup_ratio)
    elif args.scheduler == "CosineAnnealingWarmupRestarts":
        if main_process():
            logger.info("scheduler: CosineAnnealingWarmupRestarts!!!")
        scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                          first_cycle_steps=args.total_epoches * iter_per_epoch,
                                          cycle_mult=1.0,
                                          max_lr=args.learning_rate,
                                          min_lr=1e-8,
                                          warmup_steps=args.warmup_epochs*iter_per_epoch,
                                          gamma=1.0)
    else:
        raise ValueError("No such scheduler {}".format(args.scheduler))

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.5, last_epoch = init_epoch - 1)
    # LEARNING_RATE_CLIP = 1e-5

    ############################
    # start training  #
    ############################
    for epoch in range(init_epoch, args.total_epoches):
        # lr = max(optimizer.param_groups[0]['lr'], LEARNING_RATE_CLIP)
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lr
        if args.DDP:
            train_data_loader.sampler.set_epoch(epoch)

        if main_process():
            logger.info("lr: {}".format(scheduler.get_last_lr()))
        loss_train, mIoU_train, mAcc_train, allAcc_train = train(train_data_loader, model, criterion, optimizer, epoch, scheduler)

        epoch_log = epoch + 1
        if args.scheduler_update == 'epoch':
            scheduler.step()

        if main_process():
            lr = scheduler.get_last_lr()
            if isinstance(lr, list):
                lr = lr[0]
            if args.use_tensorboard:
                writer.add_scalar('loss_train', loss_train, epoch_log)
                writer.add_scalar('mIoU_train', mIoU_train, epoch_log)
                writer.add_scalar('mAcc_train', mAcc_train, epoch_log)
                writer.add_scalar('allAcc_train', allAcc_train, epoch_log)
                writer.add_scalar('lr_train', lr, epoch_log)
            else:
                metrics_dict = {'loss_train': loss_train,
                                'mIoU_train': mIoU_train,
                                'mAcc_train': mAcc_train,
                                'allAcc_train': allAcc_train,
                                'lr_train': lr}
                bolt.send_metrics(metrics_dict)

        is_best = False 
        if epoch_log % args.eval_freq == 0:
            loss_val, mIoU_val, mAcc_val, allAcc_val = validate(val_data_loader, model, val_criterion)

            if main_process():
                if args.use_tensorboard:
                    writer.add_scalar('loss_val', loss_val, epoch_log)
                    writer.add_scalar('mIoU_val', mIoU_val, epoch_log)
                    writer.add_scalar('mAcc_val', mAcc_val, epoch_log)
                    writer.add_scalar('allAcc_val', allAcc_val, epoch_log)
                else:
                    metrics_dict = {'loss_val': loss_val,
                                    'mIoU_val': mIoU_val,
                                    'mAcc_val': mAcc_val,
                                    'allAcc_val': allAcc_val}
                    bolt.send_metrics(metrics_dict)
                is_best = mIoU_val > best_iou 
                best_iou = max(best_iou, mIoU_val)

        if (epoch_log % args.save_freq == 0) and main_process():
            if not os.path.exists(args.file_dir + "/model/"):
                os.makedirs(args.file_dir + "/model/")
            filename = args.file_dir + '/model/model_last.pth'
            logger.info('Saving checkpoint to : ' + filename)
            logger.info('best IoU sofar: %.3f'%(best_iou))
            torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(), 'best_iou': best_iou, 'is_best': is_best}, filename)
            if is_best:
                shutil.copyfile(filename, args.file_dir + '/model/model_best.pth')

    if main_process():
        writer.close()
        logger.info('===>Training done!\nBest IoU: %.3f'%(best_iou))


def train(train_loader, model, criterion, optimizer, epoch, scheduler):
   
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    model.train()
    end = time.time() 
    max_iter = args.total_epoches * len(train_loader)
    if 'accum_iter' in args:
        accum_iter = args.accum_iter
    else:
        accum_iter = 1

    for i, data in enumerate(train_loader):

        features, pointclouds, edges_self, edges_forward, edges_propagate, target, norms = data
#        print('maximum points: ', max([feat.shape[0] for feat in features]))
        features, pointclouds, edges_self, edges_forward, edges_propagate, target, norms = \
            to_device(features, non_blocking=True), to_device(pointclouds, non_blocking=True), \
            to_device(edges_self, non_blocking=True), to_device(edges_forward, non_blocking = True), \
            to_device(edges_propagate, non_blocking=True), to_device(target, non_blocking=True), to_device(norms, non_blocking=True)

        data_time.update(time.time() - end)
        pred = model(features, pointclouds, edges_self, edges_forward, edges_propagate, norms)
        pred = pred.contiguous().view(-1, args.num_classes)
        target = target.view(-1, 1)[:, 0]
        loss = criterion(pred, target)
        loss /= accum_iter
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        if ((i + 1) % accum_iter == 0) or ((i+1) == len(train_loader)): 
            optimizer.step()
            optimizer.zero_grad(set_to_none=True) 


        if args.scheduler_update == 'step':
            scheduler.step()

        output = pred.max(1)[1]
        n = pred.size(0)
        loss *= n 
        count = target.new_tensor([n], dtype=torch.long)
        if args.DDP:
            dist.all_reduce(loss), dist.all_reduce(count)
        n = count.item()
        loss /= n 

        intersection, union, target = intersectionAndUnionGPU(output, target, args.num_classes, args.ignore_label)
        if args.DDP:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()

        # calculate remain time
        current_iter = epoch * len(train_loader) + i + 1
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg 
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))
        lr = scheduler.get_last_lr()
        if isinstance(lr, list):
            lr = [round(x, 8) for x in lr]
        elif isinstance(lr, float):
            lr = round(lr, 8)
        if (i + 1) % args.print_freq == 0 and main_process():
            memory = torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'Loss {loss_meter.val:.4f} '
                        'Lr: {lr} '
                        'Memory: {memory:.2f} GB '
                        'Accuracy {accuracy:.4f}.'.format(epoch+1, args.total_epoches, i + 1, len(train_loader),
                                                          batch_time=batch_time, data_time=data_time,
                                                          remain_time=remain_time,
                                                          loss_meter=loss_meter,
                                                          lr=lr,
                                                          accuracy=accuracy,
                                                          memory=memory))
            
        # add scalar to writer
        if main_process():
            if isinstance(lr, list):
                lr = lr[0]
            if args.use_tensorboard:
                writer.add_scalar('loss_train_batch', loss_meter.val, current_iter)
                writer.add_scalar('mIoU_train_batch', np.mean(intersection / (union + 1e-10)), current_iter)
                writer.add_scalar('mAcc_train_batch', np.mean(intersection / (target + 1e-10)), current_iter)
                writer.add_scalar('allAcc_train_batch', accuracy, current_iter)
                writer.add_scalar('lr_train_batch', lr, current_iter)
            else:
                metrics_dict = {'loss_train_batch': loss_meter.val,
                                'mIoU_train_batch': np.mean(intersection / (union + 1e-10)),
                                'mAcc_train_batch': np.mean(intersection / (target + 1e-10)),
                                'allAcc_train_batch': accuracy,
                                'lr_train_batch': lr}
                bolt.send_metrics(metrics_dict)
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if main_process():
        logger.info('Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(\
            epoch+1, args.total_epoches, mIoU, mAcc, allAcc))
    return loss_meter.avg, mIoU, mAcc, allAcc



def validate(val_loader, model, criterion):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>>>>>>>')
    

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    torch.cuda.empty_cache()

    model.eval()
    end = time.time()
    for i, data in enumerate(val_loader):

        features, pointclouds, edges_self, edges_forward, edges_propagate, target, norms = data
        features, pointclouds, edges_self, edges_forward, edges_propagate, target, norms = \
            to_device(features), to_device(pointclouds), \
            to_device(edges_self), to_device(edges_forward), \
            to_device(edges_propagate), to_device(target), to_device(norms)

        data_time.update(time.time() - end)

        with torch.no_grad():
            pred = model(features, pointclouds, edges_self, edges_forward, edges_propagate, norms)
            pred = pred.contiguous().view(-1, args.num_classes)
            target = target.view(-1, 1)[:, 0]
            loss = criterion(pred, target)

        output = pred.max(1)[1]
        n = output.size(0)
        loss *= n
        count = target.new_tensor([n], dtype=torch.long)
        if args.DDP:
            dist.all_reduce(loss), dist.all_reduce(count)
        n = count.item() 
        loss /= n

        intersection, union, target = intersectionAndUnionGPU(output, target, args.num_classes, args.ignore_label)
        if args.DDP:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time() 
        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        'Accuracy {accuracy:.4f}.'.format(i + 1, len(val_loader),
                                                          data_time=data_time,
                                                          batch_time=batch_time,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    # print('iou_class : ', iou_class)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if main_process():
        logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
        for i in range(args.num_classes):
            logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
    
    return loss_meter.avg, mIoU, mAcc, allAcc



if __name__ == '__main__':
    import gc 
    gc.collect()
    main()
