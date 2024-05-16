#!/bin/env python
import os
import sys
import yaml
import argparse
import logging
import math
import importlib
import datetime
import random
import munch
import time
import torch
import torch.optim as optim
import warnings
import shutil
import subprocess

import datasets
from utils.train_utils import *

def train():
    logging.info(str(args))
    metrics = args.log_metrics
    best_epoch_losses = {m: (0, 0) if m == 'f1' else (0, math.inf) for m in metrics}
    train_loss_meter = AverageValueMeter()
    val_loss_meters = {m: AverageValueMeter() for m in metrics}

    train_dataset = datasets.get_dataset(args, 'train', args.train_datasets)
    val_dataset = datasets.get_dataset(args, 'val', args.train_datasets)

    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=int(args.workers))
    dataloader_val = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=int(args.workers))
    logging.info('Length of train dataset:%d', len(train_dataset))
    logging.info('Length of test dataset:%d', len(val_dataset))

    if not args.manual_seed:
        seed = random.randint(1, 10000)
    else:
        seed = int(args.manual_seed)
    logging.info('Random Seed: %d' % seed)
    random.seed(seed)
    torch.manual_seed(seed)

    device = args.device
    model_module = importlib.import_module('.%s' % args.model_name, 'models')
    net = model_module.Model(args)
    net.to(device)
    if hasattr(model_module, 'weights_init'):
        net.apply(model_module.weights_init)
        input("here")

    lr = args.lr
    if args.lr_decay:
        if args.lr_decay_interval and args.lr_step_decay_epochs:
            raise ValueError('lr_decay_interval and lr_step_decay_epochs are mutually exclusive!')
        if args.lr_step_decay_epochs:
            decay_epoch_list = [int(ep.strip()) for ep in args.lr_step_decay_epochs.split(',')]
            decay_rate_list = [float(rt.strip()) for rt in args.lr_step_decay_rates.split(',')]

    optimizer = getattr(optim, args.optimizer)
    if args.optimizer == 'Adagrad':
        optimizer = optimizer(net.parameters(), lr=lr, initial_accumulator_value=args.initial_accum_val)
    else:
        betas = args.betas.split(',')
        betas = (float(betas[0].strip()), float(betas[1].strip()))
        optimizer = optimizer(net.parameters(), lr=lr, weight_decay=args.weight_decay, betas=betas)

    if args.load_model:
        ckpt = torch.load(args.load_model)
        net.load_state_dict(ckpt['net_state_dict'])
        logging.info("%s's previous weights loaded." % args.model_name)

    epochs_since_best_val_loss = 0
    epoch = 0 
    start_time = time.time()
    for epoch in range(args.start_epoch, args.nepoch):
        torch.cuda.empty_cache()
        train_loss_meter.reset()
        net.train()

        if args.lr_decay:
            if args.lr_decay_interval:
                if epoch > 0 and epoch % args.lr_decay_interval == 0:
                    lr = lr * args.lr_decay_rate
            elif args.lr_step_decay_epochs:
                if epoch in decay_epoch_list:
                    lr = lr * decay_rate_list[decay_epoch_list.index(epoch)]
            if args.lr_clip:
                lr = max(lr, args.lr_clip)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        for i, data in enumerate(dataloader, 0):
            optimizer.zero_grad()

            if 'dpc' in args.model_name:
                pc, ref, gt, names = data
                pc, ref, gt = pc.to(device), ref.to(device), gt.to(device)
                source, target = pc.contiguous(), ref.contiguous()
                output_dict = net(source, target, gt)
            else:
                pc, gt, labels, names = data
                pc = pc.to(device)
                gt = gt.to(device)
                labels = labels.to(device)
                inputs = pc.contiguous()
                output_dict = net(inputs, gt, labels, epoch=epoch)
            out, loss = output_dict['pred'], output_dict['loss']

            train_loss_meter.update(loss.mean().item())
            loss.backward()
            optimizer.step()

            if i % args.step_interval_to_print == 0:
                logging.info(exp_name + ' train [%d: %d/%d] loss: %f lr: %f' %
                             (epoch, i, len(train_dataset) / args.batch_size, loss.mean().item(), lr) + ' time: ' + str(time.time()-start_time)[:4] + ' track: ' + str(epochs_since_best_val_loss) )

        # if epoch % args.epoch_interval_to_save == 0:
        #     save_model('%s/network.pth' % log_dir, net)

        if epoch % args.epoch_interval_to_val == 0 or epoch == args.nepoch - 1:
            best_val_loss = val(net, epoch, val_loss_meters, dataloader_val, best_epoch_losses, device, epoch)
            if args.early_stop:
                if best_val_loss:
                    epochs_since_best_val_loss = 0
                else:
                    if epoch > args.early_stop_start:
                        epochs_since_best_val_loss += 1
                if epochs_since_best_val_loss > args.early_stop_patience:
                    print("Early stopping epoch:", epoch)
                    break

    best_val_loss = val(net, epoch, val_loss_meters, dataloader_val, best_epoch_losses, device, epoch)

    args['best_model_path'] = log_dir+'/best_network.pth'
    return


def val(net, curr_epoch_num, val_loss_meters, dataloader_val, best_epoch_losses, device, epoch):
    best_val_loss = False
    for v in val_loss_meters.values():
        v.reset()
    net.eval()

    with torch.no_grad():
        for i, data in enumerate(dataloader_val):
            if 'dpc' in args.model_name:
                pc, ref, gt, names = data
                pc, ref, gt = pc.to(device), ref.to(device), gt.to(device)
                source, target = pc.contiguous(), ref.contiguous()
                result_dict = net(source, target, gt)
            else:
                pc, gt, labels, names = data
                pc, gt, labels = pc.to(device), gt.to(device), labels.to(device)
                inputs = pc.contiguous() 
                result_dict = net(inputs, gt, labels, epoch=epoch)

            for k, v in val_loss_meters.items():
                v.update(result_dict[k].mean().item())

        fmt = 'best_%s: %f [epoch %d]; '
        best_log = ''
        for loss_type, (curr_best_epoch, curr_best_loss) in best_epoch_losses.items():
            if (val_loss_meters[loss_type].avg < curr_best_loss and loss_type != 'f1') or \
                    (val_loss_meters[loss_type].avg > curr_best_loss and loss_type == 'f1'):
                best_epoch_losses[loss_type] = (curr_epoch_num, val_loss_meters[loss_type].avg)
                best_log += fmt % (loss_type, best_epoch_losses[loss_type][1], best_epoch_losses[loss_type][0])
                if loss_type == 'loss': # or loss_type =='kld': #TODO
                    best_val_loss = True
                    save_model('%s/best_network.pth' % log_dir, net)
                    logging.info('Best net saved!')
            else:
                best_log += fmt % (loss_type, curr_best_loss, curr_best_epoch)

        curr_log = ''
        for loss_type, meter in val_loss_meters.items():
            curr_log += 'curr_%s: %f; ' % (loss_type, meter.avg)

        logging.info(curr_log)
        logging.info(best_log)
    return best_val_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train config file')
    parser.add_argument('-c', '--config', help='path to config file', required=True)
    arg = parser.parse_args()
    config_path = arg.config
    args = munch.munchify(yaml.safe_load(open(config_path)))
    
    print_time = datetime.datetime.now().isoformat()[:19]
    if args.load_model:
        exp_name = os.path.basename(os.path.dirname(args.load_model))
        log_dir = os.path.dirname(args.load_model)
    else:
        exp_name = args.model_name + '_' + args.cd_loss
        if 'base_model_name' in args:
            exp_name += '_'+args.base_model_name
        if 'encoder' in args:
            exp_name += '_'+args.encoder
            
        exp_name += '_'+print_time.replace(':',"-")
        if args.train_subset_size == None:
            print_train_subset_size = 'all'
        else:
            print_train_subset_size = str(args.train_subset_size)
        log_dir = os.path.join(args.work_dir, "_".join(args.train_datasets)+"_"+print_train_subset_size, exp_name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    print(log_dir)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(os.path.join(log_dir, 'train.log')),
                                                      logging.StreamHandler(sys.stdout)])
    
    if args.train_datasets == ['all_vertebrae']:
        args.train_datasets = ['vertebrae_C1', 'vertebrae_C2', 'vertebrae_C3', 'vertebrae_C4', 
        'vertebrae_C5', 'vertebrae_C6', 'vertebrae_C7', 'vertebrae_L1', 'vertebrae_L2', 
        'vertebrae_L3', 'vertebrae_L4', 'vertebrae_L5', 'vertebrae_T10', 'vertebrae_T11', 
        'vertebrae_T12', 'vertebrae_T1', 'vertebrae_T2', 'vertebrae_T3', 'vertebrae_T4', 
        'vertebrae_T5', 'vertebrae_T6', 'vertebrae_T7', 'vertebrae_T8', 'vertebrae_T9']

    train()

    # Update yaml in log dir
    with open(os.path.join(log_dir, os.path.basename(config_path)), 'w') as f:
        yaml.dump(args, f)
    print(os.path.join(log_dir, os.path.basename(config_path)))

    # Test
    for dataset in args.train_datasets:
        subprocess.call(['python', 'consist_test.py', '-c', os.path.join(log_dir, os.path.basename(config_path)), '-d', dataset])



