import radius_neighbors as cpp_neighbors
import numpy as np
import time
from sklearn.neighbors import KDTree
import torch


data = torch.load('test_pt.pth')
queries =data[0]
queries = np.single(queries)
for i in range(10):
#    queries = np.random.rand(50000,3)
    queries = np.single(queries)
    t1 = time.time()
    indx = cpp_neighbors.batch_kquery(queries, queries, [queries.shape[0]],[queries.shape[0]], K=int(16))
    t2 = time.time()
    print('Time for nanoflann: ', (t2 - t1)*1000, 'ms')
    print(indx[0,:])
    print(indx[1,:])
    print(indx[0:10,3])
    t3 = time.time()
    kdt = KDTree(queries)
    neighbors_idx = kdt.query(queries, k = 16, return_distance=False)
    t4 = time.time()
    print('Time for SKlearn: ', (t4-t3)*1000,'ms')
    print(neighbors_idx[0,:])
    print(neighbors_idx[1,:])
    print(neighbors_idx[0:10,3])
#    neighbors_idx = neighbors_idx[:, ::dialated_rate]



