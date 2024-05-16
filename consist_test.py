import os
import sys
import importlib
import argparse
import logging
import munch
import yaml
import numpy as np
import torch
from torchsummary import summary
import warnings
warnings.filterwarnings("ignore")

import datasets
from utils.train_utils import *

def test(test_set):
    dataset_test = datasets.get_dataset(args, test_set, args.dataset, train=False)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=int(args.workers))
    dataset_length = len(dataset_test)
    logging.info('Length of test dataset:%d', len(dataset_test))

    device = args.device
    model_module = importlib.import_module('.%s' % args.model_name, 'models')
    net = model_module.Model(args)
    net.to(device)
    net.load_state_dict(torch.load(args.best_model_path, map_location=device)['net_state_dict'])
    logging.info("%s's previous weights loaded." % args.model_name)
    net.num_mask = 0
    net.eval()
    # summary(net, [(1024,3),  (1024,3), (1024,3)])

    metrics = args.log_metrics
    test_loss_meters = {m: AverageValueMeter() for m in metrics}
    idx_to_plot = [0,1]
    
    logging.info('Testing '+test_set+'...')

    if args.save_predictions:
        pred_dir = os.path.join(log_dir, test_set)
        save_output_path = os.path.join(pred_dir, 'output')
        save_resampled_output_path = os.path.join(pred_dir, 'resampled_output')
        save_rotated_output_path = os.path.join(pred_dir, 'rotated_output')
        save_unrotated_output_path = os.path.join(pred_dir, 'unrotated_output')
        save_input_path = os.path.join(pred_dir, 'input')
        save_resampled_input_path = os.path.join(pred_dir, 'resampled_input')
        save_rotated_input_path = os.path.join(pred_dir, 'rotated_input')
        save_gt_path = os.path.join(pred_dir, 'gt')
        os.makedirs(save_output_path, exist_ok=True)
        os.makedirs(save_input_path, exist_ok=True)
        os.makedirs(save_gt_path, exist_ok=True)
        os.makedirs(save_resampled_output_path, exist_ok=True)
        # os.makedirs(save_resampled_input_path, exist_ok=True)
        # os.makedirs(save_rotated_output_path, exist_ok=True)
        os.makedirs(save_unrotated_output_path, exist_ok=True)
        # os.makedirs(save_rotated_input_path, exist_ok=True)
        if 'cae' in args.model_name:
            save_cae_output_path = os.path.join(pred_dir, 'cae_recon')
            os.makedirs(save_cae_output_path, exist_ok=True)

    with torch.no_grad():
        for i, data in enumerate(dataloader_test):
            if args.model_name[:3] == 'dpc':
                pc, ref, gt, names = data
                pc, ref, gt = pc.to(device), ref.to(device), gt.to(device)
                inputs, target = pc.contiguous(), ref.contiguous()
                result_dict = net(inputs, target, gt)
                
                random_pt_indices = torch.randint(gt.shape[1], (pc.shape[1],))
                # Resampled results
                resampled_inputs = gt[:, random_pt_indices, :]
                resampled_result_dict = net(resampled_inputs, target, gt)
                # Rotated results
                R, inv_R = get_random_rot(device=args.device)
                x_com = inputs.mean(axis=1).unsqueeze(1).repeat(1, pc.shape[1], 1)
                rotated_inputs = pc.add(-x_com) @ R 
                rotated_inputs = rotated_inputs.add(x_com)
                rotated_result_dict = net(rotated_inputs, target, gt)
                rotated_pred = rotated_result_dict['pred']
                pred_com = rotated_pred.mean(axis=1).unsqueeze(1).repeat(1, rotated_pred.shape[1], 1)
                unrotated_pred = rotated_pred.add(-pred_com) @ inv_R
                unrotated_pred = unrotated_pred.add(pred_com)
            else:
                pc, gt, labels, names = data
                pc, gt, labels = pc.to(device), gt.to(device), labels.to(device)
                inputs = pc.contiguous() 
                result_dict = net(inputs, gt, labels)
                if "4d" in args and args["4d"]:
                    random_pt_indices = torch.randint(gt.shape[2], (inputs.shape[2],))
                    # Resampled results
                    resampled_inputs = gt[:, :, random_pt_indices, :]
                    resampled_result_dict = net(resampled_inputs, gt, labels)
                    # Rotated results
                    R, inv_R = get_random_rot(device=args.device)
                    x_com = inputs.mean(axis=2).unsqueeze(2).repeat(1, 1, inputs.shape[2], 1)
                    rotated_inputs = inputs.add(-x_com) @ R 
                    rotated_inputs = rotated_inputs.add(x_com)
                    rotated_result_dict = net(rotated_inputs, gt, labels)
                    rotated_pred = rotated_result_dict['pred']
                    pred_com = rotated_pred.mean(axis=2).unsqueeze(2).repeat(1, 1, rotated_pred.shape[2], 1)
                    unrotated_pred = rotated_pred.add(-pred_com) @ inv_R
                    unrotated_pred = unrotated_pred.add(pred_com)
                else:
                    random_pt_indices = torch.randint(gt.shape[1], (inputs.shape[1],))
                    # Resampled results
                    resampled_inputs = gt[:, random_pt_indices, :]
                    resampled_result_dict = net(resampled_inputs, gt, labels)
                    # Rotated results
                    R, inv_R = get_random_rot(device=args.device)
                    x_com = inputs.mean(axis=1).unsqueeze(1).repeat(1, inputs.shape[1], 1)
                    rotated_inputs = inputs.add(-x_com) @ R 
                    rotated_inputs = rotated_inputs.add(x_com)
                    rotated_result_dict = net(rotated_inputs, gt, labels)
                    rotated_pred = rotated_result_dict['pred']
                    pred_com = rotated_pred.mean(axis=1).unsqueeze(1).repeat(1, rotated_pred.shape[1], 1)
                    unrotated_pred = rotated_pred.add(-pred_com) @ inv_R
                    unrotated_pred = unrotated_pred.add(pred_com)

            for k, v in test_loss_meters.items():
                v.update(result_dict[k].mean().item())

            if args.save_predictions:
                if "4d" in args and args["4d"]:
                    for j in range(gt.shape[0]):
                        for k in range(args.num_time_points):
                            np.savetxt(os.path.join(save_output_path, names[k][j]+'.particles'), result_dict['pred'][j][k].cpu().numpy())
                            np.savetxt(os.path.join(save_input_path, names[k][j]+'.particles'), inputs[j][k].cpu().numpy())
                            np.savetxt(os.path.join(save_gt_path, names[k][j]+'.particles'), gt[j][k].cpu().numpy())
                            np.savetxt(os.path.join(save_resampled_output_path, names[k][j]+'.particles'), resampled_result_dict['pred'][j][k].cpu().numpy())
                            np.savetxt(os.path.join(save_unrotated_output_path, names[k][j]+'.particles'), unrotated_pred[j][k].cpu().numpy())

                else:
                    for j in range(len(names)):
                        np.savetxt(os.path.join(save_output_path, names[j]+'.particles'), result_dict['pred'][j].cpu().numpy())
                        np.savetxt(os.path.join(save_input_path, names[j]+'.particles'), inputs[j].cpu().numpy())
                        np.savetxt(os.path.join(save_gt_path, names[j]+'.particles'), gt[j].cpu().numpy())
                        # np.savetxt(os.path.join(save_resampled_input_path, names[j]+'.particles'), resampled_inputs[j].cpu().numpy())
                        np.savetxt(os.path.join(save_resampled_output_path, names[j]+'.particles'), resampled_result_dict['pred'][j].cpu().numpy())
                        # np.savetxt(os.path.join(save_rotated_input_path, names[j]+'.particles'), rotated_inputs[j].cpu().numpy())
                        # np.savetxt(os.path.join(save_rotated_output_path, names[j]+'.particles'), rotated_pred[j].cpu().numpy())
                        np.savetxt(os.path.join(save_unrotated_output_path, names[j]+'.particles'), unrotated_pred[j].cpu().numpy())
                        if 'cae' in args.model_name:
                            np.savetxt(os.path.join(save_cae_output_path, names[j]+'.particles'), result_dict['cae_recon'][j].cpu().numpy())
                        
        logging.info('Overview results:')
        overview_log = ''
        for metric, meter in test_loss_meters.items():
            overview_log += '%s: %f ' % (metric, meter.avg)
        logging.info(overview_log)

def get_random_rot(deg=15, device='cuda:0'):
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
    return R.to(device), inv_R.to(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test config file')
    parser.add_argument('-c', '--config', help='path to config file', required=True)
    parser.add_argument('-d', '--dataset', help='Data class to test', required=True, default='pancreas')
    parser.add_argument('-t', '--test_set', help='train or test', default='all')
    arg = parser.parse_args()
    config_path = arg.config
    args = munch.munchify(yaml.safe_load(open(config_path)))
    if 'missing_percent' not in args:
        args['missing_percent'] = 0

    if not args.best_model_path:
        raise ValueError('Model path must be provided to load model!')

    exp_name = os.path.basename(args.best_model_path)
    print('\nTesting', arg.dataset)
    args.dataset = [arg.dataset]
    if args.dataset == ['all_vertebrae']:
        args.dataset = ['vertebrae_C1', 'vertebrae_C2', 'vertebrae_C3', 'vertebrae_C4', 
        'vertebrae_C5', 'vertebrae_C6', 'vertebrae_C7', 'vertebrae_L1', 'vertebrae_L2', 
        'vertebrae_L3', 'vertebrae_L4', 'vertebrae_L5', 'vertebrae_T10', 'vertebrae_T11', 
        'vertebrae_T12', 'vertebrae_T1', 'vertebrae_T2', 'vertebrae_T3', 'vertebrae_T4', 
        'vertebrae_T5', 'vertebrae_T6', 'vertebrae_T7', 'vertebrae_T8', 'vertebrae_T9']


    log_dir = os.path.dirname(args.best_model_path) + '/' + arg.dataset + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(os.path.join(log_dir, 'test.log')),
                                                  logging.StreamHandler(sys.stdout)])
    if arg.test_set == 'all':
        for test_set in ['train', 'val', 'test']:
            test(test_set)
    else:
        test(arg.test_set)
