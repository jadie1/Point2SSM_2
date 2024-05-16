# Point2SSM++ 4D
# COM, scaling, and consistency loss
import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import pytorch3d
import importlib

from utils.loss_utils import calc_cd, get_lightweight_mapping_error_loss
from utils.train_utils import get_random_rot
from models.encoders_4d import DGCNN_encoder, PSTnet2_encoder, RFNet_4D_encoder, CrossSectional_DGCNN_encoder
from models.point2ssm import Attention_Module

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
        self.cs_bandwidth = args.cs_bandwidth
        self.dgcnn_k = args.dgcnn_k

        self.consistency_start_epoch = args.consistency_start_epoch
        self.resample = args.sampling_invariance
        self.rotate = args.rotation_equivarince
        self.consistency_loss_weight = args.consistency_loss_weight
        self.consistency_rotation_range = args.consistency_rotation_range

        self.encoder_name = args.encoder
        if args.encoder == 'pstnet2':
            self.encoder = PSTnet2_encoder(self.latent_dim)
        elif args.encoder == 'dgcnn4d':
            self.encoder = DGCNN_encoder(self.latent_dim, self.dgcnn_k, depth=2)
        elif args.encoder == 'rfnet':
            self.encoder = RFNet_4D_encoder(c_dim=self.latent_dim,hidden_dim=self.latent_dim*4)
        elif args.encoder == 'dgcnn3d':
            self.encoder = CrossSectional_DGCNN_encoder(self.latent_dim, self.dgcnn_k, depth=2)
        else:
            print("Unimplemented encoder: " + str(args.encoder))

        intermediate_dim = min(self.latent_dim, self.num_output//2)
        self.attn_module = Attention_Module(self.latent_dim, self.num_output, intermediate_dim)

    def forward(self, x, gt, labels, epoch=0):
        B, T, N, d = x.size()
        # Center of mass alignment - set origin to 0,0,0
        com = gt.mean(axis=2)
        x_com = com.unsqueeze(2).repeat(1, 1, x.shape[2], 1)
        x = x.add(-x_com)

        ## Get x2
        # resample
        if self.resample:
            pt_indices = torch.randint(gt.shape[2], (x.shape[2],))
            x2 = gt[:, :, pt_indices, :]
        else:
            x2 = x
        # COM alignment
        x2 = x2.add(-x_com)
        # rotate
        if self.rotate:
            R, inv_R = get_random_rot(self.consistency_rotation_range)
            R, inv_R = R.to(self.device), inv_R.to(self.device)
            x2 = x2 @ R
            # get rotated gt
            gt_com = com.unsqueeze(2).repeat(1, 1, gt.shape[2], 1)
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

        features = self.encoder(full_x)        
        prob_map = self.attn_module(features.flatten(0,1))
        full_pred = prob_map @ full_x.flatten(0,1)
        full_pred = full_pred.reshape(2*B,T,self.num_output,3)

        ## Move to back to original input reference frame
        # Unscale
        full_pred = full_pred*scale
        # Undo COM alignment
        full_pred_com = torch.cat((com, com), axis=0).unsqueeze(2).repeat(1, 1, full_pred.shape[2], 1)
        full_pred = full_pred.add(full_pred_com)

        # Calculate base loss
        cd_l1, cd_l2 = calc_cd(full_pred.flatten(0,1), full_gt.flatten(0,1))
        mapping_error_loss = get_lightweight_mapping_error_loss(full_pred.flatten(0,1), full_labels.flatten(0,1), 10)
        cd_loss = cd_l2.mean()
        base_loss = cd_loss + self.mapping_error_weight*mapping_error_loss

        # Get consistency loss
        pred, pred2 = torch.split(full_pred, x.size(0))
        if self.rotate:
            # Undo rotation
            pred_com = com.unsqueeze(2).repeat(1, 1, pred2.shape[2], 1)
            un_rot_pred = pred2.add(-pred_com) @ inv_R
            pred2 = un_rot_pred.add(pred_com)
        consist_loss = F.mse_loss(pred, pred2)*pred.shape[1]

        if epoch < self.consistency_start_epoch:
            loss = base_loss
        else:
            loss = base_loss + self.consistency_loss_weight*consist_loss

        return {'pred': pred, 'loss':loss, 'base_loss':base_loss, 'consist_loss':consist_loss, 'cd_l1': cd_l1, 'cd_l2': cd_l2, 'mapping_error':mapping_error_loss}


