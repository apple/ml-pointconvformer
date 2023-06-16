import os 
import ipdb
ipdb.set_trace()

split = "test" # val, test

data_path = "/mnt/task_runtime/ScanNet/ScanNet-v2-1.0.0/data/raw"

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

print("Done!!!")




