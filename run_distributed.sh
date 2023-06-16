python -m torch.distributed.launch --nproc_per_node=$1 train_ScanNet_DDP_WarmUP.py --config ./$2
