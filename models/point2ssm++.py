# Point2SSM++
# COM, scaling, and consistency loss
import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import pytorch3d
import importlib

from models.ae import PointNet_encoder
from models.dgcnn import DGCNN_encoder
from utils.loss_utils import calc_cd
from utils.loss_utils import get_mapping_error_loss

MSE_loss = nn.MSELoss(reduction='sum')

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.num_input = args.num_input_points
        self.latent_dim = args.latent_dim
        self.num_output = args.num_output_points
        self.cd_loss = args.cd_loss
        self.device = args.device
        self.mapping_error_weight = args.mapping_error_weight

        self.consistency_start_epoch = args.consistency_start_epoch
        self.resample = args.sampling_invariance
        self.rotate = args.rotation_equivarince
        self.consistency_loss_weight = args.consistency_loss_weight
        self.consistency_rotation_range = args.consistency_rotation_range

        self.encoder_name = args.encoder
        if args.encoder == 'pn':
            self.encoder = PointNet_encoder(self.latent_dim)
        elif args.encoder == 'dgcnn':
            self.encoder = DGCNN_encoder(self.latent_dim)
        else:
            print("Unimplemented encoder: " + str(args.encoder))

        self.attn_module = Attention_Module(self.latent_dim, self.num_output)

    def get_random_rot(self, deg=15):
        deg = torch.deg2rad(torch.tensor(deg))
        theta_x = torch.distributions.Uniform(low=-1*deg, high=deg).sample()
        theta_y = torch.distributions.Uniform(low=-1*deg, high=deg).sample()
        theta_z = torch.distributions.Uniform(low=-1*deg, high=deg).sample()
        R1 = torch.eye(3)
        R1[1, 1] = torch.cos(theta_x)
        R1[2, 2] = torch.cos(theta_x)
        R1[1, 2] = -1*torch.sin(theta_x)
        R1[2, 1] = torch.sin(theta_x)
        R2 = torch.eye(3)
        R2[0, 0] = torch.cos(theta_y)
        R2[2, 2] = torch.cos(theta_y)
        R2[2, 0] = -1*torch.sin(theta_y)
        R2[0, 2] = torch.sin(theta_y)
        R3 = torch.eye(3)
        R3[1, 1] = torch.cos(theta_z)
        R3[2, 2] = torch.cos(theta_z)
        R3[1, 2] = -1*torch.sin(theta_z)
        R3[2, 1] = torch.sin(theta_z)
        R =torch.matmul(torch.matmul(R1, R2), R3)
        inv_R = torch.linalg.inv(R)
        return R.to(self.device), inv_R.to(self.device)

    def forward(self, x, gt, labels, epoch=0):
        # Center of mass alignment - set origin to 0,0,0
        com = gt.mean(axis=1)
        x_com = com.unsqueeze(1).repeat(1, x.shape[1], 1)
        x = x.add(-x_com)

        ## Get x2
        # resample
        if self.resample:
            pt_indices = torch.randint(gt.shape[1], (x.shape[1],))
            x2 = gt[:, pt_indices, :]
        else:
            x2 = x
        # COM alignment
        x2 = x2.add(-x_com)
        # rotate
        if self.rotate:
            R, inv_R = self.get_random_rot(self.consistency_rotation_range)
            x2 = x2 @ R
            # get rotated gt
            gt_com = com.unsqueeze(1).repeat(1, gt.shape[1], 1)
            gt2 = gt.add(-gt_com) @ R
            gt2 = gt2.add(gt_com)
        else:
            gt2 = gt

        # Double batch
        full_x = torch.cat((x, x2), axis=0)
        full_gt = torch.cat((gt, gt2), axis=0)
        full_labels = torch.cat((labels,labels), axis=0)

        # Scale by max distance of pts to origin
        scale, _ = full_x.norm(dim=2, keepdim=True).max(dim=1, keepdim=True)
        full_x = full_x/scale

        # Encode
        z, features = self.encoder(full_x)
        if self.encoder_name == 'dgcnn':
            features = features.transpose(2,1)

        # Attention Modula
        prob_map = self.attn_module(features)

         # Output
        full_pred = torch.sum(prob_map[:, :, :, None] * full_x[:, None, :, :], dim=2)

        ## Move to back to original input reference frame
        # Unscale
        full_pred = full_pred*scale
        # Undo COM alignment
        full_pred_com = torch.cat((com, com), axis=0).unsqueeze(1).repeat(1, full_pred.shape[1], 1)
        full_pred = full_pred.add(full_pred_com)

        # Calculate base loss
        cd_l1, cd_l2 = calc_cd(full_pred, full_gt)
        mapping_error_loss = get_mapping_error_loss(full_pred, full_labels, 10)
        if self.cd_loss == 'cd_l1':
            cd_loss = cd_l1.mean()
        elif self.cd_loss == 'cd_l2':
            cd_loss = cd_l2.mean()
        base_loss = cd_loss + self.mapping_error_weight*mapping_error_loss

        # Get consistency loss
        pred, pred2 = torch.split(full_pred, x.size(0))
        if self.rotate:
            # Undo rotation
            pred_com = com.unsqueeze(1).repeat(1, pred2.shape[1], 1)
            un_rot_pred = pred2.add(-pred_com) @ inv_R
            pred2 = un_rot_pred.add(pred_com)
        consist_loss = F.mse_loss(pred, pred2)*pred.shape[1]

        if epoch < self.consistency_start_epoch:
            loss = base_loss
        else:
            loss = base_loss + self.consistency_loss_weight*consist_loss

        return {'pred': pred, 'loss':loss, 'base_loss':base_loss, 'consist_loss':consist_loss, 'cd_l1': cd_l1, 'cd_l2': cd_l2, 'mapping_error':mapping_error_loss}


class Attention_Module(nn.Module):
    def __init__(self, latent_dim, num_output):
        super(Attention_Module, self).__init__()
        self.num_output = num_output
        self.latent_dim = latent_dim

        self.sa1 = cross_transformer(self.latent_dim,self.num_output)
        self.sa2 = cross_transformer(self.num_output,self.num_output)
        self.sa3 = cross_transformer(self.num_output,self.num_output)
        self.softmax = nn.Softmax(dim=2)        

    def forward(self, x):
        x = self.sa1(x,x)
        x = self.sa2(x,x)
        x = self.sa3(x,x)
        prob_map = self.softmax(x)
        return prob_map


# PointAttN: You Only Need Attention for Point Cloud Completion
# https://github.com/ohhhyeahhh/PointAttN
class cross_transformer(nn.Module):
    def __init__(self, d_model=256, d_model_out=256, nhead=4, dim_feedforward=1024, dropout=0.0):
        super().__init__()
        self.multihead_attn1 = nn.MultiheadAttention(d_model_out, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear11 = nn.Linear(d_model_out, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear12 = nn.Linear(dim_feedforward, d_model_out)

        self.norm12 = nn.LayerNorm(d_model_out)
        self.norm13 = nn.LayerNorm(d_model_out)

        self.dropout12 = nn.Dropout(dropout)
        self.dropout13 = nn.Dropout(dropout)

        self.activation1 = torch.nn.GELU()

        self.input_proj = nn.Conv1d(d_model, d_model_out, kernel_size=1)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    # transformer
    def forward(self, src1, src2, if_act=False):
        src1 = self.input_proj(src1)
        src2 = self.input_proj(src2)

        b, c, _ = src1.shape

        src1 = src1.reshape(b, c, -1).permute(2, 0, 1)
        src2 = src2.reshape(b, c, -1).permute(2, 0, 1)

        src1 = self.norm13(src1)
        src2 = self.norm13(src2)

        src12 = self.multihead_attn1(query=src1,
                                     key=src2,
                                     value=src2)[0]

        src1 = src1 + self.dropout12(src12)
        src1 = self.norm12(src1)

        src12 = self.linear12(self.dropout1(self.activation1(self.linear11(src1))))
        src1 = src1 + self.dropout13(src12)

        src1 = src1.permute(1, 2, 0)

        return src1