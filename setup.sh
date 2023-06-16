#!/bin/bash

apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
add-apt-repository ppa:deadsnakes/ppa
apt update
apt install -y zip libglu1-mesa
apt install python3.8-distutils


FILE=./data
if [ ! -d "$FILE" ]; then
    echo "mkdir $FILE"
    mkdir $FILE
fi 

#unzip ScanNet_withNewNormal.zip
#rm ScanNet_withNewNormal.zip
#cd ../

pip install -U ipdb scikit-learn matplotlib open3d easydict

pip install tensorboard timm termcolor tensorboardX

cd cpp_wrappers/
sh compile_wrappers.sh
cd ..






