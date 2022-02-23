#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Training script for TearingNet
'''

import torch
import torch.nn.parallel
import torch.optim as optim
import sys
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))

# neural atlas
import yaml
import easydict
import pytorch_lightning as pl
import neural_atlas as neat

from dataloaders import point_cloud_dataset_train, gen_rotation_matrix
from models.autoencoder import PointCloudAutoencoder
from models.tearingnet import TearingNetBasicModel
from models.tearingnet_graph import TearingNetGraphModel
from util.option_handler import TrainOptionHandler
from torch.utils.tensorboard import SummaryWriter
import time

import warnings
warnings.filterwarnings("ignore")

def main():
    # load the neural atlas config from the config file
    with open(opt.neat_config) as f:
        neat_config = easydict.EasyDict(yaml.full_load(f))

    # set configs from the neural atlas config
    assert neat_config.data.train_eff_batch_size \
           == neat_config.data.val_eff_batch_size

    # seed all pseudo-random generators
    neat_config.seed = pl.seed_everything(neat_config.seed, workers=True)

    # instantiate the neural atlas data module
    datamodule = neat.data.datamodule.DataModule(
        neat_config.seed,
        neat_config.input,
        neat_config.target,
        num_nodes=1,
        gpus=[ 0 ],
        **neat_config.data
    )
    datamodule.setup(stage="fit")
    train_dataloader = datamodule.train_dataloader()['dataset']
    opt.svr = ("img" in neat_config.input)

    # Build up an autoencoder
    t = time.time()
    print(torch.cuda.device_count(), "GPUs will be used for training.")
    device = torch.device("cuda:0")
    ae = PointCloudAutoencoder(opt)
    ae = torch.nn.DataParallel(ae)
    ae.to(device)
    tearingnet_basic = isinstance(ae.module.decoder, TearingNetBasicModel)
    tearingnet_graph = isinstance(ae.module.decoder, TearingNetGraphModel)

    # Create folder to save trained models
    if not os.path.exists(opt.exp_name):
        os.makedirs(opt.exp_name)

    # Create a tensorboard writer
    if opt.tf_summary: writer = SummaryWriter(log_dir=opt.exp_name)

    # Setup the optimizer and the scheduler
    lr = opt.lr
    optimizer = optim.Adam(ae.parameters(), lr=lr[0])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr[1], gamma=lr[2], last_epoch=-1)

    # Load a check point if given
    if opt.checkpoint != '':
        checkpoint = torch.load(opt.checkpoint)
        if opt.load_weight_only == False:
            ae.module.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Existing model %s fully loaded.\n" % (opt.checkpoint))
        else:
            model_dict = ae.module.state_dict() # load parameters from pre-trained FoldingNet
            for k in checkpoint['model_state_dict']:
                if k in model_dict:
                    model_dict[k] = checkpoint['model_state_dict'][k]
                    print("    Found weight: " + k)
                elif k.replace('folding1', 'folding') in model_dict:
                    model_dict[k.replace('folding1', 'folding')] = checkpoint['model_state_dict'][k]
                    print("    Found weight: " + k)
            ae.module.load_state_dict(model_dict)
            print("Existing model %s (partly) loaded.\n" % (opt.checkpoint))
    epoch_start = scheduler.state_dict()['last_epoch']

    # Start training
    n_iter = 0
    for epoch in range(epoch_start, opt.n_epoch):
        ae.train()
        print('Training at epoch %d with lr %f' % (epoch, scheduler.get_lr()[0]))

        train_loss_xyz_sum = 0
        if tearingnet_basic:
            train_loss_xyz_pre_sum = 0
        if tearingnet_graph:
            train_loss_xyz_pre_sum = 0
            train_loss_xyz_pre2_sum = 0
        train_loss_sum = 0
        batch_id = 0
        not_end_yet = True
        it = iter(train_dataloader)
        if not opt.svr:
            len_train = len(train_dataloader)
        else:
            len_train = len(train_dataloader) // 24

        # Iterates the training process
        while not_end_yet == True:
            batch = easydict.EasyDict(next(it))
            if not opt.svr:
                batch.input.pcl = batch.input.pcl.cuda()
                input = batch.input.pcl
            else:
                batch.input.img = batch.input.img.cuda()
                input = batch.input.img
            batch.target.pcl = batch.target.pcl.cuda()

            not_end_yet = batch_id + 1 < len_train
            optimizer.zero_grad()

            # Forward and backward
            rec = ae(input)
            grid = rec['grid']
            if tearingnet_basic:
                rec_pre = rec['rec_pre']
            if tearingnet_graph:
                rec_pre = rec['rec_pre']
                rec_pre2 = rec['rec_pre2']
            rec = rec['rec']

            # Forward and backward computation
            train_loss_xyz = ae.module.xyz_loss(batch.target.pcl, rec)
            if tearingnet_basic:
                train_loss_xyz_pre = ae.module.xyz_loss(batch.target.pcl, rec_pre)
            if tearingnet_graph:
                train_loss_xyz_pre = ae.module.xyz_loss(batch.target.pcl, rec_pre)
                train_loss_xyz_pre2 = ae.module.xyz_loss(batch.target.pcl, rec_pre2)
            train_loss = train_loss_xyz # may add other loss if needed
            train_loss.backward()
            optimizer.step()

            # Log the result
            train_loss_xyz_sum += train_loss_xyz.item()
            if tearingnet_basic: train_loss_xyz_pre_sum += train_loss_xyz_pre.item()
            if tearingnet_graph:
                train_loss_xyz_pre_sum += train_loss_xyz_pre.item()
                train_loss_xyz_pre2_sum += train_loss_xyz_pre2.item()
            train_loss_sum += train_loss.item()
            if batch_id % opt.print_freq == 0:
                elapse = time.time() - t
                if tearingnet_basic: 
                    print('    batch_no: %d/%d, time/iter: %f/%d, loss_xyz: %f, loss_xyz_pre: %f, loss: %f' % 
                        (batch_id, len_train, elapse, n_iter, train_loss_xyz.item(), train_loss_xyz_pre.item(), train_loss.item()))
                elif tearingnet_graph:
                    print('    batch_no: %d/%d, time/iter: %f/%d, loss_xyz: %f, loss_xyz_pre: %f, loss_xyz_pre2: %f, loss: %f' % 
                        (batch_id, len_train, elapse, n_iter, train_loss_xyz.item(), train_loss_xyz_pre.item(), train_loss_xyz_pre2.item(), train_loss.item()))
                else:
                    print('    batch_no: %d/%d, time/iter: %f/%d, loss_xyz: %f, loss: %f' % 
                        (batch_id, len_train, elapse, n_iter, train_loss_xyz.item(), train_loss.item()))
                if opt.tf_summary: writer.add_scalar('train/batch_train_loss', train_loss.item(), n_iter)

            if opt.pc_write_freq > 0 and batch_id % opt.pc_write_freq == 0:
                labels = np.concatenate((np.ones(rec.shape[1]), np.zeros(batch.target.pcl.shape[1])), axis=0).tolist()
                if opt.tf_summary: 
                    writer.add_embedding(torch.cat((rec[0, 0:, 0:3], batch.target.pcl[0, :, 0:3]), 0), 
                        global_step=n_iter, metadata=labels, tag="pc") # the output point cloud
                    writer.add_embedding(grid[0, 0:, 0:2], global_step=n_iter, tag="grid") # the 2D grid being used
            n_iter = n_iter + 1
            batch_id = batch_id + 1
        
        # Output the average loss of current epoch
        avg_loss_xyz = train_loss_xyz_sum / (batch_id + 1)
        if tearingnet_basic:
            avg_loss_xyz_pre = train_loss_xyz_pre_sum / (batch_id + 1)
        if tearingnet_graph:
            avg_loss_xyz_pre = train_loss_xyz_pre_sum / (batch_id + 1)
            avg_loss_xyz_pre2 = train_loss_xyz_pre2_sum / (batch_id + 1)
        avg_loss = train_loss_sum / (batch_id + 1)

        elapse = time.time() - t
        if tearingnet_basic:
            print('Epoch: %d    time: %f --- avg_loss_xyz: %f, avg_loss_xyz_pre: %f, avg_loss: %f    lr: %f' % \
                (epoch, elapse, avg_loss_xyz, avg_loss_xyz_pre, avg_loss, scheduler.get_lr()[0]))
        elif tearingnet_graph:
            print('Epoch: %d    time: %f --- avg_loss_xyz: %f, avg_loss_xyz_pre: %f, avg_loss_xyz_pre2: %f, avg_loss: %f    lr: %f' % \
                (epoch, elapse, avg_loss_xyz, avg_loss_xyz_pre, avg_loss_xyz_pre2, avg_loss, scheduler.get_lr()[0]))
        else:
            print('Epoch: %d    time: %f --- avg_loss_xyz: %f, avg_loss: %f    lr: %f' % \
                (epoch, elapse, avg_loss_xyz, avg_loss, scheduler.get_lr()[0]))
        if opt.tf_summary: writer.add_scalar('train/epoch_train_loss', avg_loss, epoch)
        if opt.tf_summary: writer.add_scalar('train/learning_rate', scheduler.get_lr()[0], epoch)
        scheduler.step()

        # Save the checkpoint
        if epoch % opt.save_epoch_freq == 0 or epoch == opt.n_epoch - 1:
            dict_name=opt.exp_name + '/epoch_'+str(epoch)+'.pth'
            torch.save({
                'model_state_dict': ae.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'last_epoch': scheduler.state_dict()['last_epoch'],
                'opt': opt
            }, dict_name)
            print('Current checkpoint saved to %s.' % (dict_name))
        print('\n')

    if opt.tf_summary: writer.close()
    print('Done!')


if __name__ == "__main__":

    option_handler = TrainOptionHandler()
    opt = option_handler.parse_options() # all options are parsed through this command
    option_handler.print_options(opt) # print out all the options
    main()
