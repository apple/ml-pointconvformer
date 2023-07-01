# PointConvFormer
*Wenxuan Wu, Li Fuxin, Qi Shan*

This is the PyTorch implementation of our paper [**PointConvFormer**]
<div align="center">
    <img src="figs/pcf.jpg">
</div>

## Introduction

We introduce PointConvFormer, a novel building block for point cloud based deep network architectures. Inspired
by generalization theory, PointConvFormer combines ideas from point convolution, where filter weights are only based
on relative position, and Transformers which utilize feature-based attention. In PointConvFormer, attention computed
from feature difference between points in the neighborhood is used to modify the convolutional weights at each point.
Hence, we preserved the invariances from point convolution, whereas attention helps to select relevant points in the
neighborhood for convolution. PointConvFormer is suitable for multiple tasks that require details at the point level, such
as segmentation and scene flow estimation tasks. Our results show that PointConvFormer offers a better accuracy-speed
tradeoff than classic convolutions, regular transformers, and voxelized sparse convolution approaches. Visualizations
show that PointConvFormer performs similarly to convolution on flat areas, whereas the neighborhood selection effect is stronger on object boundaries, showing that it has got the best of both worlds.

## Highlight
1. We introduce PointConvFormer which modifies convolution by an attention weight computed from the  differences of local neighbourhood features. We further extend the PointConvFormer with a multi-head mechanism.
2. We conduct thorough experiments on semantic segmentation tasks for both indoor and outdoor scenes, as well as  scene flow estimation from 3D point clouds on multiple datasets. Extensive ablation studies are conducted to study the properties and design choice of PointConvFormer.

## Citation
Please cite it as W. Wu, L. Fuxin, Q. Shan. PointConvFormer: Revenge of the Point-Cloud Convolution. CVPR 2023

## Installation

### Environment
The minimal GPU requirement is GTX 1050 due to the use of CUDA compute capabilities 6.

1. Install dependencies

One would need a working PyTorch installed. This code was tested to work in pyTorch 1.8 and 1.12, with CUDA 10.1 and 11.3 respectively.

One possibility is to run: 
```
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.1 -c pytorch
```
However usually you should already have an environment that takes care of pyTorch.

Afterwards, running setup.sh will download all the prepared ScanNet dataset as well as installing the required packages. 

```
setup.sh
```

### Data Preparation

### Training

1. Before training, please setup the `train_data_path` and `val_data_path` in `configPCF_10cm.yaml`;

2. You might also want to set the `model_name`, `experiment_dir` accordingly in `configPCF_10cm.yaml`;

4. Change other settings in `configFLPCF_10cm.yaml` based on your experiments;

5. Make sure you have the same number of CPUs as num_workers in the config file (e.g. configPCF_10cm.yaml).

6. Run ```sh run_distributed.sh num_gpus config_file_name``` to train the model with num_gpus GPUs, or python train_ScanNet_DDP_WarmUP.py --config configPCF_10cm.yaml to train the model with a single GPU.

### Evaluation

<!-- (Obselete Please download the pretrain weights of the models at [here](https://drive.google.com/file/d/1BShjM0PydlEX-bE7k3-fg2UBORpwUeWR/view?usp=sharing)) -->

Then, you can evaluate with the following comand:

```
python test_ScanNet_simple.py --config ./configPCF_10cm.yaml --pretrain_path ./pretrain/[model_weights].pth
```
which will evaluate the time as well as outputting ply files for each scene.

### Configuration Files and Default Models

We have provided a few sample configuration files for the ScanNet dataset. configPCF_10cm.yaml is a sample 10cm configuration file
and configPCF_10cm_lite.yaml corresponds to the lightweight version in the paper. Similar configuration files were provided for the 5cm
configuration and a configuration file configPCF_2cm_PTF2.yaml is also provided utilizing some of the ideas from PointTransformerv2
(grid size, mix3D augmentation), which is cheaper than the main one reported in the paper yet still achieves a comparable 74.4% mIOU on the ScanNet validation set.

Besides, in model_architecture.py we have provided a few default models such as PCF_Tiny, PCF_Small, PCF_Normal and PCF_Large, which can serve for different scenarios.
