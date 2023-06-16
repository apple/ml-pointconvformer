# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import glob, plyfile, numpy as np, multiprocessing as mp, torch
from plyfile import PlyData, PlyElement
import open3d as o3d

# Map relevant classes to {0,1,...,19}, and ignored classes to -100
remapper=np.ones(150)*(-100)
for i,x in enumerate([1,2,3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39]):
    remapper[x]=i

basepath = "/nfs/stak/users/wuwen/hpc-share/dataset/scannet"

files=sorted(glob.glob(os.path.join(basepath, '*/*_vh_clean_2.ply')))
files2=sorted(glob.glob(os.path.join(basepath, '*/*_vh_clean_2.labels.ply')))
assert len(files) == len(files2)

def load_obj_with_normals(filepath):
    mesh = o3d.io.read_triangle_mesh(str(filepath))
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    coords = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)
    colors = np.asarray(mesh.vertex_colors)
    feats = np.hstack((colors, normals))

    return coords, feats

def newf(fn):
    # import ipdb; ipdb.set_trace()
    fn2 = fn[:-3]+'labels.ply'
    coords, feats = load_obj_with_normals(fn)
    a=plyfile.PlyData().read(fn2)
    labels=remapper[np.array(a.elements[0]['label'])]
    torch.save((coords,feats,labels, fn.split('/')[-1]), fn[:-4]+'.pth')
    print(fn, fn2)


p = mp.Pool(processes=mp.cpu_count())
p.map(newf,files)
p.close()
p.join()



