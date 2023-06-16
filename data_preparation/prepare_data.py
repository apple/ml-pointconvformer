# Prepare ScanNet Data into the form we want

import os
import glob
import plyfile
import numpy as np
import multiprocessing as mp
import torch
import open3d as o3d
import argparse


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
    fn2 = fn[:-3]+'labels.ply'
    coords, feats = load_obj_with_normals(fn)
    a = plyfile.PlyData().read(fn2)
    labels = remapper[np.array(a.elements[0]['label'])]
    torch.save((coords, feats, labels, fn.split('/')[-1]), fn[:-4]+'.pth')
    print(fn, fn2)


def newf_test(fn):
    coords, feats = load_obj_with_normals(fn)
    num_points = coords.shape[0]
    labels = np.zeros(num_points).astype(np.int32)
    torch.save((coords, feats, labels, fn.split('/')[-1]), fn[:-4] + '.pth')
    print(fn)

if __name__ == '__main__':
    # Map relevant classes to {0,1,...,19}, and ignored classes to -100
    parser = argparse.ArgumentParser(description="Prepare ScanNet Data")
    parser.add_argument("basepath", help="base path of the downloaded ScanNet dataset")
    parser.add_argument("split", help="The split that you want to parse (train/validation/test). For test, the labels are set to 0")
    args = parser.parse_args()
    data_path = args.basepath
    split = args.split
    if split != "test":
        scan_path = os.path.join(data_path, "scans")
    else:
        scan_path = os.path.join(data_path, "scans_test")

    with open("scannetv2_" + split + ".txt") as f:
        scans_name = [x.strip() for x in f.readlines()]

    print(len(scans_name))

    for scan_name in scans_name:
        print("copy scene: ", scan_name)
        ply_path = os.path.join(scan_path, scan_name, scan_name + "_vh_clean_2.ply")
        label_path = os.path.join(scan_path, scan_name, scan_name + "_vh_clean_2.labels.ply")
        segsjson_path = os.path.join(scan_path, scan_name, scan_name + "_vh_clean_2.0.010000.segs.json")
        aggjson_path = os.path.join(scan_path, scan_name, scan_name + ".aggregation.json")

        os.system('cp %s %s' % (ply_path, split))

        if split != "test":
            os.system('cp %s %s' % (label_path, split))
            os.system('cp %s %s' % (segsjson_path, split))
            os.system('cp %s %s' % (aggjson_path, split))

    remapper = np.ones(150)*(-100)
    for i, x in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]):
        remapper[x] = i

    files = sorted(glob.glob(os.path.join(data_path, '*/*_vh_clean_2.ply')))
    if split != 'test':
        files2 = sorted(glob.glob(os.path.join(data_path, '*/*_vh_clean_2.labels.ply')))
        assert len(files) == len(files2)
    p = mp.Pool(processes=mp.cpu_count())
    if split != 'test':
        p.map(newf, files)
    else:
        p.map(newf_test, files)
    p.close()
    p.join()
