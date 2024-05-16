import os
import sys
import math
import torch
import pytorch3d
import numpy as np
import torch.linalg
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d import loss
from torch_cluster import knn
import itertools
import random

def calc_cd(output, gt):
    cd_l1, _ = pytorch3d.loss.chamfer_distance(output, gt, norm=1, point_reduction='sum')
    cd_l2, _ = pytorch3d.loss.chamfer_distance(output, gt, norm=2, point_reduction='sum')
    return cd_l1, cd_l2

def repeat_calc_cd(pred, gt):
    save_pred = pred
    pred = torch.cat(gt.shape[1]//pred.shape[1]*[save_pred], axis=1)
    pred = torch.cat([pred, save_pred[:,:gt.shape[1]%pred.shape[1], :]], axis=1) # remainder
    cd_l1, _ = pytorch3d.loss.chamfer_distance(pred, gt, norm=1, point_reduction='sum')
    cd_l2, _ = pytorch3d.loss.chamfer_distance(pred, gt, norm=2, point_reduction='sum')
    return cd_l1, cd_l2

# Source: https://github.com/dvirginz/DPC
def get_mapping_error_loss(pred, labels, k=10):
    edge_index = [knn(pred[i], pred[i], k,) for i in range(pred.shape[0])]
    neigh_idxs = torch.stack([edge_index[i][1].reshape(pred.shape[1], -1) for i in range(pred.shape[0])])
    batch_size = pred.shape[0]
    loss, count = 0, 0
    for source_index in range(batch_size):
        for target_index in range(batch_size):
            if source_index != target_index:
                if labels[source_index] == labels[target_index]:
                    count += 1
                    loss += mapping_error_loss_helper(pred[source_index].unsqueeze(0), neigh_idxs[source_index].unsqueeze(0), pred[target_index].unsqueeze(0), k)
    if count == 0:
        return 0
    else: 
        return (loss/count).mean()

# Source: https://github.com/dvirginz/DPC
def get_lightweight_mapping_error_loss(pred, labels, k=10, num_pairs=4):
    edge_index = [knn(pred[i], pred[i], k,) for i in range(pred.shape[0])]
    neigh_idxs = torch.stack([edge_index[i][1].reshape(pred.shape[1], -1) for i in range(pred.shape[0])])
    batch_size = pred.shape[0]
    loss, count = 0, 0
    pairs = list(itertools.combinations(list(range(batch_size)), 2))
    random.shuffle(pairs) 
    for pair in pairs[:num_pairs]:
        source_index, target_index = pair
        if source_index != target_index:
            if labels[source_index] == labels[target_index]:
                count += 1
                loss += mapping_error_loss_helper(pred[source_index].unsqueeze(0), neigh_idxs[source_index].unsqueeze(0), pred[target_index].unsqueeze(0), k)
    if count == 0:
        return 0
    else: 
        return (loss/count).mean()

# Source: https://github.com/dvirginz/DPC
def mapping_error_loss_helper(source, source_neighs, target, k):
    # Source: (1, N, 3) Source neighs: (1, N, k), target: (1, N, 3)
    # source_grouped = grouping_operation(source.transpose(1, 2).contiguous(), source_neighs.int()).permute(0, 2, 3, 1) # (1, N, K, 3)
    source_grouped = pytorch3d.ops.knn_gather(source.contiguous(), source_neighs)
    source_diff = source_grouped[:, :, 1:, :] - torch.unsqueeze(source, 2)  # remove fist grouped element, as it is the seed point itself
    source_square = torch.sum(source_diff ** 2, dim=-1)

    # target_cr_grouped = grouping_operation(target.transpose(1, 2).contiguous(), source_neighs.int()).permute(0, 2, 3, 1)
    target_cr_grouped = pytorch3d.ops.knn_gather(target.contiguous(), source_neighs)
    target_cr_diff = target_cr_grouped[:, :, 1:, :] - torch.unsqueeze(target, 2)  # remove fist grouped element, as it is the seed point itself
    target_cr_square = torch.sum(target_cr_diff ** 2, dim=-1)

    GAUSSIAN_HEAT_KERNEL_T = 8.0
    gaussian_heat_kernel = torch.exp(-source_square/GAUSSIAN_HEAT_KERNEL_T)
    neighbor_loss_per_neigh = torch.mul(gaussian_heat_kernel, target_cr_square)

    neighbor_loss = torch.sum(neighbor_loss_per_neigh)

    return neighbor_loss


# Source: "Learning Canonical Embeddings for Unsupervised Shape Correspondence with Locally Linear Transformations"
#   https://arxiv.org/pdf/2209.02152.pdf
# reconstructed target and target point coordinates <-- B X N x 3 and B X N x 3
# bandwidth, the kernel bandwith <-- scalar
def gmm(rec_target, target, bandwidth):
    # B X N x N x 3 <-- B X N X 1 x C - B X 1 X N X C
    diff_ij = (rec_target.unsqueeze(2) - target.unsqueeze(1))
    # B X N x N
    factor = 2*bandwidth*bandwidth
    # B X N x N
    diff_ij = (diff_ij**2).sum(-1).div(factor).mul(-0.5) -0.5*math.log(2*math.pi) - math.log(math.sqrt(2)*bandwidth)
    dist = torch.logsumexp((diff_ij).reshape(diff_ij.shape[0], -1),dim=1).mean()
    return dist

def cs_divergence(rec_target, target, bandwidth):
    target, _ = pytorch3d.ops.sample_farthest_points(target, K=rec_target.shape[1])
    r_t_dist = -1 * gmm(rec_target, target, bandwidth)
    r_r_dist = 0.5 * gmm(rec_target, rec_target, bandwidth)
    t_t_dist = 0.5 * gmm(target, target, bandwidth)
    return r_t_dist + r_r_dist + t_t_dist

def repeat_cs_divergence(pred, target, bandwidth):
    save_pred = pred
    pred = torch.cat(target.shape[1]//pred.shape[1]*[save_pred], axis=1)
    rec_target = torch.cat([pred, save_pred[:,:target.shape[1]%pred.shape[1], :]], axis=1) # remainder
    r_t_dist = -1 * gmm(rec_target, target, bandwidth)
    r_r_dist = 0.5 * gmm(rec_target, rec_target, bandwidth)
    t_t_dist = 0.5 * gmm(target, target, bandwidth)
    return r_t_dist + r_r_dist + t_t_dist


def get_pose_loss(pred, rot_pred, R, device='cuda:0'):
    gt = R[None,:,:].repeat(pred.size(0), 1, 1) # batch
    mat = batch_compute_similarity_transform_torch(pred, rot_pred)

    # # Debug
    # est_rot_pred = pred @ mat
    # np.savetxt('debug/est_rot_output.particles', est_rot_pred[0].detach().cpu().numpy())

    frob = torch.sqrt(torch.sum(torch.square(mat - gt)))    # Forbunius Norm
    angle_ = torch.mean(torch.arcsin(
        torch.clamp(torch.min(torch.tensor(1.).to(device), frob / (2. * torch.sqrt(torch.tensor(2.).to(device)))), -0.99999,
                    0.99999)))
    return angle_
 
# Ref: SC3K
def batch_compute_similarity_transform_torch(S1, S2):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.

    help: https://gist.github.com/mkocabas/54ea2ff3b03260e3fedf8ad22536f427

    '''
    # transposed = False
    # if S1.shape[0] != 3 and S1.shape[0] != 2:
    S1 = S1.permute(0,2,1)
    S2 = S2.permute(0,2,1)
    transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=-1, keepdims=True)
    mu2 = S2.mean(axis=-1, keepdims=True)

    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1**2, dim=1).sum(dim=1)

    # 3. The outer product of X1 and X2.
    K = X1.bmm(X2.permute(0,2,1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, V = torch.svd(K)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
    Z = Z.repeat(U.shape[0],1,1)
    Z[:,-1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0,2,1))))

    # Construct R.
    R = V.bmm(Z.bmm(U.permute(0,2,1)))      # position
    R = torch.linalg.inv(R)                 # rotation
    return R

# https://github.com/IIT-PAVIS/SC3K/blob/main/utils.py
def get_separation_loss(kp):
    '''
    Parameters
    ----------
    kp:         Key-points
    Method:     compute distances of each point from all the points in "kp"
                consider minimum two distances (distance of a point form itself (distance==0) and the next closest (distance>0))
                take mean of the distances from the closest point (distance>0)

    Returns     separation loss ->  average distance of every point from closest points
    -------
    '''
    min_distances = torch.cat([torch.squeeze(
        torch.norm(kp[i].unsqueeze(1) - kp[i].unsqueeze(0), dim=2, p=None).topk(2, largest=False, dim=0)[
            0]) for i in range(len(kp))], dim=0)

    return 1/torch.mean(min_distances[min_distances>0])

# https://github.com/IIT-PAVIS/SC3K/blob/main/utils.py
def get_overlap_loss(kp, threshold=0.05):
    '''
    Parameters
    ----------
    kp:         Key-points
    threshold   allowable overlap between the key-points
    Method:     Find distance of every point from all the points
                select the minimum distances that are greater than 0 (distance from itself)
                return count of the separated distances => final loss

    Returns     separation loss -> avoid estimation of multiple key-points on the same 3D location
    -------
    '''

    distances = torch.cat([torch.squeeze(
        torch.norm(kp[i].unsqueeze(1) - kp[i].unsqueeze(0), dim=2, p=None)) for i in range(len(kp))], dim=0)

    return torch.sum(distances[(distances < threshold)] >0).float() / len(distances)*len(distances)

# https://github.com/IIT-PAVIS/SC3K/blob/main/utils.py
def get_shape_loss(pc, kp):
    '''
    Parameters
    ----------
    pc      Input point cloud
    kp      Estimated key-points

    Returns Shape loss -> how far the key-points are estimated from the input point cloud
    -------

    '''
    loss = torch.cat([torch.squeeze(
        torch.norm(pc[i].unsqueeze(1) - kp[i].unsqueeze(0), dim=2, p=None).topk(1, largest=False, dim=0)[
            0]) for i in range(len(kp))], dim=0)
    return torch.mean(loss)

# https://github.com/IIT-PAVIS/SC3K/blob/main/utils.py
def get_volume_loss(kp, pc):
    '''

    Parameters: 3D IoU loss
                => same as coverage loss of clara's Paper
                => https://github.com/cfernandezlab/Category-Specific-Keypoints/blob/master/models/losses.py
    Smooth L1 loss: https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html#torch.nn.SmoothL1Loss
    ----------
    kp: Estimated key-points [BxNx3]
    pc: Point cloud [Bx2048x3]

    Returns: Int value -> IoU b/w kp and pc
    -------

    '''
    val_max_pc, _ = torch.max(pc, 1)    # Bx3
    val_min_pc, _ = torch.min(pc, 1)    # Bx3
    dim_pc = val_max_pc - val_min_pc    # Bx3
    val_max_kp, _ = torch.max(kp, 1)    # Bx3
    val_min_kp, _ = torch.min(kp, 1)    # Bx3
    dim_kp = val_max_kp - val_min_kp    # Bx3

    return F.smooth_l1_loss(dim_kp, dim_pc)

  
# import torch
# from torch_cluster import knn
# import os
# import sys
# import math
# from utils.thin_plate_spline import ThinPlateSpline
# from utils.model_utils import calc_cd
# import numpy as np
# import torch.linalg
# import torch.nn as nn
# import torch.nn.functional as F


# # from model_utils import calc_cd
# proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(os.path.join(proj_dir, "utils/Pointnet2.PyTorch/pointnet2"))
# from pointnet2_utils import grouping_operation
# import pointnet2_cuda as pointnet2

# def furthest_point_downsampling(points, npoint):
#     xyz = points.contiguous()
#     B, N, _ = xyz.size()
#     output = torch.IntTensor(B, npoint).to(xyz.device)
#     temp = torch.FloatTensor(B, N).fill_(1e10).to(xyz.device)
#     pointnet2.furthest_point_sampling_wrapper(B, N, npoint, xyz, temp, output)
#     indices = output.cpu().numpy()
#     subset = []
#     for i in range(points.shape[0]):
#         subset.append(points[i][indices[i], :].cpu().numpy())
#     subset = np.array(subset)
#     return torch.FloatTensor(subset).to(points.device)







# # all combos within batch
# def get_tps_loss(pred, inpt, reg=0.0):
#     batch_size = pred.shape[0]
#     loss, count = 0, 0
#     for source_index in range(batch_size):
#         for target_index in range(batch_size):
#             if source_index != target_index:
#                 count += 1
#                 loss += tps_loss_helper(pred[source_index], inpt[source_index], pred[target_index], inpt[target_index], reg)
#     return (loss/(count/2)).mean()

# def tps_loss_helper(C_a, PC_a, C_b, PC_b, reg):
#     tps = ThinPlateSpline(alpha=reg, device=C_a.device)
#     tps.fit(C_a, C_b)
#     recon_b = tps.transform(PC_a)
#     cd_p, cd_t = calc_cd(PC_b.unsqueeze(0), recon_b.unsqueeze(0))    
#     tps.fit(C_b, C_a)
#     recon_a = tps.transform(PC_b)
#     np.savetxt('debug/PC_a.particles',PC_a.detach().cpu().numpy())
#     np.savetxt('debug/PC_b.particles',PC_b.detach().cpu().numpy())
#     np.savetxt('debug/C_a.particles',C_a.detach().cpu().numpy())
#     np.savetxt('debug/C_b.particles',C_b.detach().cpu().numpy())
#     np.savetxt('debug/recon_a.particles',recon_a.detach().cpu().numpy())
#     np.savetxt('debug/recon_b.particles', recon_b.detach().cpu().numpy())
#     return cd_t.mean()

# # https://github.com/IIT-PAVIS/SC3K/blob/main/utils.py
# def get_consistency_loss(pred, rot_pred, R):
#     un_rot_pred = rot_pred @ torch.linalg.inv(R)
#     return F.mse_loss(pred, un_rot_pred)