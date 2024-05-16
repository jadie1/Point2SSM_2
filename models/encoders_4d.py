import torch
import torch.nn as nn
import torch_cluster
from models.dgcnn import DGCNN_encoder as DGCNN_encoder_3d

# For PSTnet2
import os 
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from pst_operations import *

# 4D encoders take sequences of point clouds as input for forward with dimension: (batch_size x number of time points x number of points x dim of points (3))
# They output 4D features with shape (batch size x number of time points x feature size x number of points)

# Based off of https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py
# DGCNN encoder adapted to pool along temporal dimension to get global feature which is concatentated to all features within a sequence
class DGCNN_encoder(nn.Module):
    def __init__(self, latent_dim, k=27, depth=4, extra_features=0):
        super(DGCNN_encoder, self).__init__()
        self.num_neighs = k
        self.latent_dim = latent_dim
        self.input_features = (3+extra_features) * 2
        self.only_true_neighs = True
        self.depth = depth
        bb_size = 24
        output_dim = self.latent_dim # 768
        last_in_dim = 0
        self.convs = []
        for i in range(self.depth):
            in_features = self.input_features if i == 0 else out_features*4
            out_features = bb_size * 2 if i == 0 else in_features
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_features, out_features, kernel_size=1, bias=False), nn.BatchNorm2d(out_features), nn.LeakyReLU(negative_slope=0.2),
                )
            )
            # last_in_dim =+ out_features*2
        last_in_dim = bb_size * 2 * sum([(4 ** i)//2 for i in range(1,self.depth + 1,1)])
        self.convs.append(
            nn.Sequential(
                nn.Conv1d(last_in_dim, self.latent_dim, kernel_size=1, bias=False), nn.BatchNorm1d(self.latent_dim), nn.LeakyReLU(negative_slope=0.2),
            )
        )
        self.convs = nn.ModuleList(self.convs)

    def forward_per_point(self, x, start_neighs=None):
        self.num_points = x.shape[1]
        x = x.transpose(1, 2)  # DGCNN assumes BxFxN

        if(start_neighs is None):
            start_neighs = torch_cluster.knn(x,k=self.num_neighs)
        x = get_graph_feature(x, k=self.num_neighs, idx=start_neighs, only_intrinsic=False)#only_intrinsic=self.hparams.only_intrinsic)

        outs = [x]
        for conv in self.convs[:-1]:
            if(len(outs) > 1):
                x = get_graph_feature(outs[-1], k=self.num_neighs, idx=None if not self.only_true_neighs else start_neighs)
            x = conv(x)
            x_max = x.max(dim=-1, keepdim=False)[0] # BT, F, N
            expanded_x = x_max.reshape((self.batch_size, self.time_steps, x_max.shape[1], x_max.shape[2])) # B, T, F, N
            temporal_pooled = avgpool(expanded_x, dim=1, keepdim=True).expand(expanded_x.size()) # B, T, F, N
            combined = torch.cat([expanded_x,temporal_pooled], dim=2) # B, T, 2F, N
            flat_combined = combined.flatten(0,1) # BT, 2F, N
            outs.append(flat_combined)

        x = torch.cat(outs[1:], dim=1)
        features = self.convs[-1](x)
        return features

    def forward(self, x, return_neighs=False):
        self.batch_size, self.time_steps, self.num_points, dim = x.shape
        flat_x = x.flatten(0,1)
        sigmoid_for_classification=True
        edge_index = [
            torch_cluster.knn(flat_x[i,:,:3], flat_x[i,:,:3], self.num_neighs,)
            for i in range(flat_x.shape[0])
        ]
        neigh_idx = torch.stack(
            [edge_index[i][1].reshape(flat_x.shape[1], -1) for i in range(flat_x.shape[0])]
        )
        features_per_point = self.forward_per_point(flat_x, start_neighs=neigh_idx) # BT, F, N
        features = features_per_point.reshape(self.batch_size, self.time_steps, features_per_point.shape[1], features_per_point.shape[2]) # B, T, F, N
        return features

# DGCNN encoder helper
def get_graph_feature(x, k, idx=None, only_intrinsic=False, permute_feature=True):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    else:
        if(len(idx.shape)==2):
            idx = idx.unsqueeze(0).repeat(batch_size,1,1)
        idx = idx[:, :, :k]
        k = min(k,idx.shape[-1])

    num_idx = idx.shape[1]

    idx_base = torch.arange(0, batch_size, device=idx.device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.contiguous()
    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(
        2, 1
    ).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_idx, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    if only_intrinsic is True:
        feature = feature - x
    elif only_intrinsic == 'neighs':
        feature = feature
    elif only_intrinsic == 'concat':
        feature = torch.cat((feature, x), dim=3)
    else:
        feature = torch.cat((feature - x, x), dim=3)

    if permute_feature:
        feature = feature.permute(0, 3, 1, 2).contiguous()

    return feature

# Source: https://github.com/hkust-vgd/RFNet-4D/blob/main/im2mesh/encoder/pointnet.py
class RFNet_4D_encoder(nn.Module):
    def __init__(self,
                 c_dim=128,
                 dim=3,
                 hidden_dim=512,
                 use_only_first_pcl=False,
                 **kwargs):
        super().__init__()
        self.c_dim = c_dim
        self.use_only_first_pcl = use_only_first_pcl

        self.spatial_fc_pos = nn.Linear(dim, 2 * hidden_dim)
        self.spatial_block_0 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.spatial_block_1 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.spatial_block_2 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.spatial_block_3 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.spatial_block_4 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.spatial_fc_c = nn.Linear(hidden_dim, c_dim)

        self.temporal_fc_pos = nn.Linear(dim + 1, 3 * hidden_dim)
        self.temporal_block_0 = ResnetBlockFC(3 * hidden_dim, hidden_dim)
        self.temporal_block_1 = ResnetBlockFC(3 * hidden_dim, hidden_dim)
        self.temporal_block_2 = ResnetBlockFC(3 * hidden_dim, hidden_dim)
        self.temporal_block_3 = ResnetBlockFC(3 * hidden_dim, hidden_dim)
        self.temporal_block_4 = ResnetBlockFC(3 * hidden_dim, hidden_dim)
        # self.temporal_fc_c = nn.Linear(2 * hidden_dim, c_dim)
        self.temporal_fc_c = nn.Linear(hidden_dim, c_dim)

        self.fc_c = nn.Linear(2 * c_dim, c_dim)

        self.actvn = nn.ReLU()
        self.pool = maxpool

    def forward(self, x):
        batch_size, n_steps, n_pts, n_dim = x.shape
        t = (torch.arange(n_steps, dtype=torch.float32) / (n_steps - 1)).to(
            x.device)
        t = t[None, :, None, None].expand(batch_size, n_steps, n_pts, 1)
        x_t = torch.cat([x, t], dim=3).reshape(batch_size, n_steps, n_pts,
                                               n_dim + 1)

        # Spatial
        net = self.spatial_fc_pos(x)
        net = self.spatial_block_0(net)
        pooled = self.pool(net, dim=2, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=3)

        net = self.spatial_block_1(net)
        pooled = self.pool(net, dim=2, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=3)

        net = self.spatial_block_2(net)
        pooled = self.pool(net, dim=2, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=3)

        net = self.spatial_block_3(net)
        pooled = self.pool(net, dim=2, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=3)

        net = self.spatial_block_4(net) #batch_size x n_steps x input_pts x hidden_dim
        # net = self.pool(net, dim=2) #batch_size x n_steps x hidden_dim
        spatial_c = self.spatial_fc_c(
            self.actvn(net))  #batch_size x n_steps x input_pts x hidden_dim

        # Temporal
        net = self.temporal_fc_pos(x_t)
        net = self.temporal_block_0(net)
        pooled = self.pool(net, dim=2, keepdim=True)
        pooled_expand = pooled.expand(net.size())
        pooled2 = self.pool(pooled, dim=1, keepdim=True)
        pooled2_expand = pooled2.expand(net.size())
        net = torch.cat([net, pooled_expand, pooled2_expand], dim=3)

        net = self.temporal_block_1(net)
        pooled = self.pool(net, dim=2, keepdim=True)
        pooled_expand = pooled.expand(net.size())
        pooled2 = self.pool(pooled, dim=1, keepdim=True)
        pooled2_expand = pooled2.expand(net.size())
        net = torch.cat([net, pooled_expand, pooled2_expand], dim=3)

        net = self.temporal_block_2(net)
        pooled = self.pool(net, dim=2, keepdim=True)
        pooled_expand = pooled.expand(net.size())
        pooled2 = self.pool(pooled, dim=1, keepdim=True)
        pooled2_expand = pooled2.expand(net.size())
        net = torch.cat([net, pooled_expand, pooled2_expand], dim=3)

        net = self.temporal_block_3(net)
        pooled = self.pool(net, dim=2, keepdim=True)
        pooled_expand = pooled.expand(net.size())
        pooled2 = self.pool(pooled, dim=1, keepdim=True)
        pooled2_expand = pooled2.expand(net.size())
        net = torch.cat([net, pooled_expand, pooled2_expand], dim=3)

        net = self.temporal_block_4(net) #batch_size x n_steps x input_pts x hidden_dim
        # net = self.pool(net, dim=2) #batch_size x n_steps x hidden_dim
        # pooled = self.pool(net, dim=1, keepdim=True).expand(net.size()) #batch_size x n_steps x hidden_dim
        # net = torch.cat([net, pooled], dim=2) #batch_size x n_steps x 2*hidden_dim
        temporal_c = self.temporal_fc_c(
            self.actvn(net))  #batch_size x n_steps x input_pts x hidden_dim

        # spatiotemporal_c = torch.cat([spatial_c, temporal_c], dim=2)
        spatiotemporal_c = torch.cat([spatial_c, temporal_c], dim=3)
        spatiotemporal_c = self.fc_c(
            self.actvn(spatiotemporal_c))
        return spatiotemporal_c.transpose(-2,-1) # batch_size x n_steps x c_dim x num_points

# https://github.com/hkust-vgd/RFNet-4D/blob/main/im2mesh/encoder/pointnet.py
def maxpool(x, dim=-1, keepdim=False):
    ''' Performs a maxpooling operation.

    Args:
        x (tensor): input
        dim (int): dimension of pooling
        keepdim (bool): whether to keep dimensions
    '''
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out

def avgpool(x, dim=-1, keepdim=False):
    ''' Performs a maxpooling operation.

    Args:
        x (tensor): input
        dim (int): dimension of pooling
        keepdim (bool): whether to keep dimensions
    '''
    out, _ = x.mean(dim=dim, keepdim=keepdim)
    return out

# https://github.com/hkust-vgd/RFNet-4D/blob/main/im2mesh/layers.py
# RFNet_4D_Encoder layers - Resnet Blocks
class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''
    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out

        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)

        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx

# Based off of https://github.com/hehefan/PSTNet2
class PSTnet2_encoder(nn.Module):
    def __init__(self, latent_dim, radius=0.25, nsamples=16):
        super(PSTnet2_encoder, self).__init__()
        self.latent_dim = latent_dim

        self.encoder1  = PSTOp(in_channels=3,
                               spatial_radius=radius,
                               spatial_neighbours=nsamples,
                               spatial_sampling=4,
                               spatial_channels=[self.latent_dim//4, self.latent_dim//4, self.latent_dim],
                               spatial_batch_norm=[True, True, True],
                               spatial_activation=[True, True, True],
                               temporal_radius=0,
                               temporal_stride=1,
                               temporal_padding=[0,0],
                               temporal_padding_mode="replicate",
                               temporal_channels=[],
                               temporal_batch_norm=[],
                               temporal_activation=[])

        self.encoder2  = PSTOp(in_channels=self.latent_dim,
                               spatial_radius=2*radius,
                               spatial_neighbours=nsamples,
                               spatial_sampling=4,
                               spatial_channels=[self.latent_dim//2, self.latent_dim//2, self.latent_dim*2],
                               spatial_batch_norm=[True, True, True],
                               spatial_activation=[True, True, True],
                               temporal_radius=0,
                               temporal_stride=1,
                               temporal_padding=[0,0],
                               temporal_padding_mode="replicate",
                               temporal_channels=[],
                               temporal_batch_norm=[],
                               temporal_activation=[])

        self.encoder3  = PSTOp(in_channels=self.latent_dim*2,
                               spatial_radius=4*radius,
                               spatial_neighbours=nsamples,
                               spatial_sampling=4,
                               spatial_channels=[self.latent_dim, self.latent_dim, self.latent_dim*4],
                               spatial_batch_norm=[True, True, True],
                               spatial_activation=[True, True, True],
                               temporal_radius=0,
                               temporal_stride=1,
                               temporal_padding=[0,0],
                               temporal_padding_mode="replicate",
                               temporal_channels=[],
                               temporal_batch_norm=[],
                               temporal_activation=[])

        self.encoder4  = PSTOp(in_channels=self.latent_dim*4,
                               spatial_radius=8*radius,
                               spatial_neighbours=nsamples,
                               spatial_sampling=2,
                               spatial_channels=[self.latent_dim*2, self.latent_dim*2, self.latent_dim*8],
                               spatial_batch_norm=[True, True, True],
                               spatial_activation=[True, True, True],
                               temporal_radius=0,
                               temporal_stride=1,
                               temporal_padding=[0,0],
                               temporal_padding_mode="replicate",
                               temporal_channels=[],
                               temporal_batch_norm=[],
                               temporal_activation=[])


        self.decoder4 = PSTTransOp(in_channels=self.latent_dim*8,
                                   temporal_radius=0,
                                   temporal_stride=1,
                                   temporal_padding=[0, 0],
                                   temporal_channels=[],
                                   temporal_batch_norm=[],
                                   temporal_activation=[],
                                   spatial_channels=[self.latent_dim*2, self.latent_dim*2],
                                   spatial_batch_norm=[True, True],
                                   spatial_activation=[True, True],
                                   original_channels=self.latent_dim*4)

        self.decoder3 = PSTTransOp(in_channels=self.latent_dim*2,
                                   temporal_radius=0,
                                   temporal_stride=1,
                                   temporal_padding=[0, 0],
                                   temporal_channels=[],
                                   temporal_batch_norm=[],
                                   temporal_activation=[],
                                   spatial_channels=[self.latent_dim*2, self.latent_dim*2],
                                   spatial_batch_norm=[True, True],
                                   spatial_activation=[True, True],
                                   original_channels=self.latent_dim*2)

        self.decoder2 = PSTTransOp(in_channels=self.latent_dim*2,
                                   temporal_radius=0,
                                   temporal_stride=1,
                                   temporal_padding=[0, 0],
                                   temporal_channels=[],
                                   temporal_batch_norm=[],
                                   temporal_activation=[],
                                   spatial_channels=[self.latent_dim*2, self.latent_dim],
                                   spatial_batch_norm=[True, True],
                                   spatial_activation=[True, True],
                                   original_channels=self.latent_dim)

        self.decoder1 = PSTTransOp(in_channels=self.latent_dim,
                                   temporal_radius=0,
                                   temporal_stride=1,
                                   temporal_padding=[0, 0],
                                   temporal_channels=[],
                                   temporal_batch_norm=[],
                                   temporal_activation=[],
                                   spatial_channels=[self.latent_dim, self.latent_dim],
                                   spatial_batch_norm=[True, True],
                                   spatial_activation=[True, True],
                                   original_channels=3)


    def forward(self, xyzs):

        new_xyzs1, new_features1 = self.encoder1(xyzs, xyzs.transpose(2,3))

        new_xyzs2, new_features2 = self.encoder2(new_xyzs1, new_features1)

        new_xyzs3, new_features3 = self.encoder3(new_xyzs2, new_features2)

        new_xyzs4, new_features4 = self.encoder4(new_xyzs3, new_features3)


        new_xyzsd4, new_featuresd4 = self.decoder4(new_xyzs4, new_xyzs3, new_features4, new_features3)

        new_xyzsd3, new_featuresd3 = self.decoder3(new_xyzsd4, new_xyzs2, new_featuresd4, new_features2)

        new_xyzsd2, new_featuresd2 = self.decoder2(new_xyzsd3, new_xyzs1, new_featuresd3, new_features1)

        new_xyzsd1, new_featuresd1 = self.decoder1(new_xyzsd2, xyzs, new_featuresd2, xyzs.transpose(2,3))


        # out = self.outconv(new_featuresd1.transpose(1,2)).transpose(1,2)

        return new_featuresd1

class CrossSectional_DGCNN_encoder(nn.Module):
    def __init__(self, latent_dim, k=27, depth=4, bb_size=12, extra_features=0):
        super(CrossSectional_DGCNN_encoder, self).__init__()
        self.encoder = DGCNN_encoder_3d(latent_dim, k, depth, bb_size=bb_size)
    def forward(self, x):
        self.batch_size, self.time_steps, self.num_points, dim = x.shape
        flat_x = x.flatten(0,1)
        z, features = self.encoder(flat_x)
        features = features.transpose(1,2)
        return  features.reshape(self.batch_size, self.time_steps, features.shape[1], features.shape[2]) # B, T, F, N



