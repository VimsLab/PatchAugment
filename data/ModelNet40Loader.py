import torch
import torch.utils.data as data
import numpy as np
import os, sys, h5py
from numpy.random import choice
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

def _get_data_files(list_filename):
    with open(list_filename) as f:
        return [line.rstrip()[5:] for line in f]

def _load_data_file(name):
    f = h5py.File(name)
    data = f['data'][:]
    label = f['label'][:]
    return data, label

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

class ModelNet40Cls(data.Dataset):

    def __init__(
            self, num_points, root, transforms=None, train=True
    ):
        super().__init__()

        self.transforms = transforms

        root = os.path.abspath(root)
        self.folder = "modelnet40_ply_hdf5_2048"
        self.data_dir = os.path.join(root, self.folder)

        self.train, self.num_points = train, num_points
        if self.train:
            self.files =  _get_data_files( \
                os.path.join(self.data_dir, 'train_files.txt'))
        else:
            self.files =  _get_data_files( \
                os.path.join(self.data_dir, 'test_files.txt'))

        point_list, label_list = [], []
        # self.point_fname = []
        for f in self.files:
            points, labels = _load_data_file(os.path.join(root, f))
            point_list.append(points)
            label_list.append(labels)

            # self.point_fname.append(f)

        self.points = np.concatenate(point_list, 0)
        self.labels = np.concatenate(label_list, 0)

    def __getitem__(self, idx):
        pt_idxs = np.arange(0, self.num_points)   # 2048
        if self.train:
            np.random.shuffle(pt_idxs)

        current_points = self.points[idx, pt_idxs].copy()
        # print(current_points.dtype)
        label = torch.from_numpy(self.labels[idx]).type(torch.LongTensor)
        # print(len(self.point_fname),label)
        # print(self.labels[idx][0])
        # if self.labels[idx][0] == 17:
        #     print('F')
        #     fig = plt.figure()
        #     ax = fig.add_subplot(111, projection='3d')
        #     ax.scatter(current_points[:, 0], current_points[:, 1], current_points[:, 2],
        #                color='red',
        #                edgecolor='black',
        #                s=5,
        #                alpha=1)
        #     plt.show()

        if self.transforms is not None:
            current_points = self.transforms(current_points)
        # print(current_points.shape)
        # exit()
        # current_points = farthest_point_sample(current_points, self.num_points)

        return current_points, label

    def __len__(self):
        return self.points.shape[0]

if __name__ == "__main__":
    from torchvision import transforms
    import data_utils as d_utils

    transforms = transforms.Compose([
        d_utils.PointcloudToTensor(),
        # d_utils.PointcloudRotate(axis=np.array([1,0,0])),
        # d_utils.PointcloudScale(),
        # d_utils.PointcloudTranslate(),
        # d_utils.PointcloudJitter()
    ])
    dset = ModelNet40Cls(16, "./", train=True, transforms=transforms)
    print(dset[0][0])
    print(dset[0][1])
    print(len(dset))
    dloader = torch.utils.data.DataLoader(dset, batch_size=32, shuffle=True)
