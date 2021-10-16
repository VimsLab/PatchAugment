import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import cos, sin
from time import time
import numpy as np
from sanity_check import check_Input

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def knn(x, k):
    '''
    get k nearest neighbors' indices for a single point cloud feature
    :param x:  x is point cloud feature, shape: [B, F, N]
    :param k:  k is the number of neighbors
    :return: KNN graph, shape: [B, N, k]
    '''
    # print(x.shape)
    inner = -2*torch.matmul(x, x.transpose(2, 1))
    # print(inner.shape)
    xx = torch.sum(x**2, dim=2, keepdim=True)
    # print(xx.shape)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    # print(pairwise_distance.shape)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    # print(idx.shape)
    return idx

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    torch.cuda.empty_cache()
    new_xyz = index_points(xyz, fps_idx)
    torch.cuda.empty_cache()
    idx = knn(xyz, nsample)#query_ball_point(radius, nsample, xyz, new_xyz)
    torch.cuda.empty_cache()
    idx2 = index_points(idx,fps_idx)
    torch.cuda.empty_cache()
    grouped_xyz = index_points(xyz, idx2) # [B, npoint, nsample, C]
    torch.cuda.empty_cache()
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    torch.cuda.empty_cache()

    if points is not None:
        grouped_points = index_points(points, idx2)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points

def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

class Augmentor(nn.Module):
    def __init__(self, npoint, nsample):
        super(Augmentor, self).__init__()
        self.npoint = npoint
        self.nsample = nsample*3//4
        self.scale_factor = nn.Parameter(torch.FloatTensor(self.npoint, 1), requires_grad=False)
        self.translation_factor = nn.Parameter(torch.FloatTensor(self.npoint, 1), requires_grad=False)
        self.jitter_factor = nn.Parameter(torch.FloatTensor(self.npoint, self.nsample,  3), requires_grad=False)
        self.theta = nn.Parameter(torch.FloatTensor(self.npoint, 1), requires_grad=False)
        self.alpha = nn.Parameter(torch.FloatTensor(self.npoint, 1), requires_grad=False)
        self.beta = nn.Parameter(torch.FloatTensor(self.npoint, 1), requires_grad=False)
        self.gamma = nn.Parameter(torch.FloatTensor(self.npoint, 1), requires_grad=False)
        self.initalize()

    def initalize(self):
        #initialing the PatchAugment parameters
        torch.nn.init.uniform_(self.theta, a=-0.1, b=0.1)
        torch.nn.init.uniform_(self.alpha, a=-0.1, b=0.1)
        torch.nn.init.uniform_(self.beta, a=-0.1, b=0.1)
        torch.nn.init.uniform_(self.gamma, a=-0.1, b=0.1)
        torch.nn.init.uniform_(self.scale_factor, a=0.95, b=1.05)
        torch.nn.init.uniform_(self.translation_factor, a=-0.05, b=0.05)
        torch.nn.init.uniform_(self.jitter_factor, a=-0.01, b=0.01)

    def forward(self, new_points):
        bs = new_points.shape[0]
        drop_idx=torch.randperm(self.nsample)[:self.nsample]
        new_points_x = new_points[:, :, drop_idx, 0]
        new_points_y = new_points[:, :, drop_idx, 1]
        new_points_z = new_points[:, :, drop_idx, 2]

        # new_points_x_n = new_points[:, :, :, 3]
        # new_points_y_n = new_points[:, :, :, 4]
        # new_points_z_n = new_points[:, :, :, 5]

        new_points_x *= self.scale_factor
        new_points_y *= self.scale_factor
        new_points_z *= self.scale_factor

        #save after scale
        # print(self.scale_factor[0])
        # s_new_points = torch.stack([new_points_x, new_points_y, new_points_z], 3)

        #Rotation
        a = (cos(torch.sqrt(self.alpha**2))*cos(torch.sqrt(self.beta**2))).cuda()
        b = (cos(torch.sqrt(self.alpha**2))*sin(torch.sqrt(self.beta**2))*sin(torch.sqrt(self.gamma**2))-sin(torch.sqrt(self.alpha**2))*cos(torch.sqrt(self.gamma**2))).cuda()
        c = (cos(torch.sqrt(self.alpha**2))*sin(torch.sqrt(self.beta**2))*cos(torch.sqrt(self.gamma**2))+sin(torch.sqrt(self.alpha**2))*sin(torch.sqrt(self.gamma**2))).cuda()
        d = (sin(torch.sqrt(self.alpha**2))*cos(torch.sqrt(self.beta**2))).cuda()
        e = (sin(torch.sqrt(self.alpha**2))*sin(torch.sqrt(self.beta**2))*sin(torch.sqrt(self.gamma**2))+cos(torch.sqrt(self.alpha**2))*cos(torch.sqrt(self.gamma**2))).cuda()
        f = (sin(torch.sqrt(self.alpha**2))*sin(torch.sqrt(self.beta**2))*cos(torch.sqrt(self.gamma**2))-cos(torch.sqrt(self.alpha**2))*sin(torch.sqrt(self.gamma**2))).cuda()
        g = (-sin(torch.sqrt(self.beta**2))).cuda()
        h = (cos(torch.sqrt(self.beta**2))*sin(torch.sqrt(self.gamma**2))).cuda()
        i = (cos(torch.sqrt(self.beta**2))*cos(torch.sqrt(self.gamma**2))).cuda()
        # print(a,b,c,d,e,f,g,h,i)
        new_points_x_r = new_points_x * a + new_points_y * b + new_points_z * c
        new_points_y_r = new_points_x * d + new_points_y * e + new_points_z * f
        new_points_z_r = new_points_x * g + new_points_y * h + new_points_z * i

        # new_points_x_r_n = new_points_x_n * a + new_points_y_n * b + new_points_z_n * c
        # new_points_y_r_n = new_points_x_n * d + new_points_y_n * e + new_points_z_n * f
        # new_points_z_r_n = new_points_x_n * g + new_points_y_n * h + new_points_z_n * i

        #rotate along y-axis
        aa = (cos(torch.sqrt(self.theta**2))).cuda()
        bb = (torch.zeros(self.npoint, 1)).cuda()
        cc = (sin(torch.sqrt(self.theta**2))).cuda()
        dd = (torch.ones(self.npoint,1)).cuda()

        new_points_x_rr = new_points_x_r * aa + new_points_y_r * bb + new_points_z_r * cc
        new_points_y_rr = new_points_x_r * bb + new_points_y_r * dd + new_points_z_r * bb
        new_points_z_rr = -new_points_x_r * cc + new_points_y_r * bb + new_points_z_r * aa

        # new_points_x_rr_n = new_points_x_r_n * aa + new_points_y_r_n * bb + new_points_z_r_n * cc
        # new_points_y_rr_n = new_points_x_r_n * bb + new_points_y_r_n * dd + new_points_z_r_n * bb
        # new_points_z_rr_n = -new_points_x_r_n * cc + new_points_y_r_n * bb + new_points_z_r_n * aa
        #save after rotate
        # r_new_points = torch.stack([new_points_x_r, new_points_y_r, new_points_z_r], 3)
        # normals_n = torch.stack([new_points_x_rr_n, new_points_y_rr_n, new_points_z_rr_n],3)
        #Translate
        new_points_x = new_points_x_rr + self.translation_factor
        new_points_y = new_points_y_rr + self.translation_factor
        new_points_z = new_points_z_rr + self.translation_factor

        #save after Translate
        new_xyz_norm = torch.stack([new_points_x, new_points_y, new_points_z], 3)
        # t_new_points = new_xyz_norm

        #save after jitter
        new_xyz_norm[:, :, :, :] += self.jitter_factor#[:, 0:new_xyz_norm.shape[2],:]

        #point drop_point
        # nsample = self.nsample*3//4
        # drop_idx=torch.randperm(nsample)[:nsample]
        # new_xyz_norm = new_xyz_norm[:,:,drop_idx,:]
        # assert new_xyz_norm.shape == (bs, self.npoint, nsample, 3)

        new_points = new_points[:,:,drop_idx,:]
        # assert new_points.shape == (bs, self.npoint, nsample, new_points.shape[3])
        # normals_n = normals_n[:,:,drop_idx,:]
        new_points = torch.cat([new_xyz_norm, new_points[:, :, :, 3:]], 3)
        # print(new_points.shape)
        # j_new_points = new_points

        # check_Input(new_points, s_new_points, r_new_points, t_new_points, j_new_points)

        return new_points

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all
        # Augmentor
        if self.npoint is not None:
            self.augmentor = Augmentor(self.npoint, self.nsample)


    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """

        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
            new_points = self.augmentor(new_points)

        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            # print(new_points.shape)
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        self.augmentors = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)
            if self.npoint is not None and self.training:
                self.augmentors.append(Augmentor(self.npoint, self.nsample_list[i]))

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz
            if self.training:
                grouped_points = self.augmentors[i](grouped_points)
            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points
