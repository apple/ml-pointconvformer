#!/bin/bash

echo "\nalias blobby='aws --endpoint-url https://blob.mr3.simcloud.apple.com --cli-read-timeout 300'" >> ~/.bashrc
. ~/.bashrc

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

cd $FILE
blobby s3 cp s3://wenxuan_wu/ScanNet_withNewNormal.zip ./

unzip ScanNet_withNewNormal.zip
rm ScanNet_withNewNormal.zip
cd ../

pip install -U ipdb scikit-learn matplotlib open3d easydict
pip install --upgrade turibolt --index https://pypi.apple.com/simple

# conda install pytorch torchvision torchaudio -c pytorch
# conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=11.3 -c pytorch -c conda-forge

#conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.1 -c pytorch

pip install tensorboard timm termcolor tensorboardX

cd cpp_wrappers/
sh compile_wrappers.sh
cd ..

# rm semanticKitti_normal.zip

# cd ../

# pip install -U ipdb scikit-learn matplotlib open3d

# cp semantic-kitti.yaml ./data/dataset/






