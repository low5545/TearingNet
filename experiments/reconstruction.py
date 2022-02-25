#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Reconstruction experiment
'''

import open3d as o3d
import torch
import sys
import os
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))

# neural atlas
import time
import yaml
import easydict
import tqdm
import pytorch_lightning as pl
import neural_atlas as neat
from torch.utils.tensorboard import SummaryWriter

from dataloaders import point_cloud_dataset_test
from models.autoencoder import PointCloudAutoencoder
from util.option_handler import TestOptionHandler
from util.mesh_writer import write_ply_mesh

import warnings
warnings.filterwarnings("ignore")

def main():
    # load the neural atlas config from the config file
    with open(opt.neat_config) as f:
        neat_config = easydict.EasyDict(yaml.full_load(f))

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
    datamodule.setup(stage="test")
    test_dataset = datamodule.test_dataset
    test_dataloader = datamodule.test_dataloader().loaders['dataset']
    opt.svr = ("img" in neat_config.input)

    print(torch.cuda.device_count(), "GPUs will be used for testing.")

    # Load a saved model
    if opt.checkpoint == '':
        print("Please provide the model path.")
        exit()
    else:
        checkpoint = torch.load(opt.checkpoint)
        if 'opt' in checkpoint and opt.config_from_checkpoint == True:
            checkpoint['opt'].grid_dims = opt.grid_dims
            checkpoint['opt'].xyz_chamfer_weight = opt.xyz_chamfer_weight
            
            checkpoint['opt'].graph_eps = opt.graph_delete_point_eps
            if "svr" not in checkpoint['opt']:
                checkpoint['opt'].svr = opt.svr

            ae = PointCloudAutoencoder(checkpoint['opt'])
            print("\nModel configuration loaded from checkpoint %s." % opt.checkpoint)
        else:
            ae = PointCloudAutoencoder(opt)
            checkpoint['opt'] = opt
            torch.save(checkpoint, opt.checkpoint)
            print("\nModel configuration written to checkpoint %s." % opt.checkpoint)
        ae.load_state_dict(checkpoint['model_state_dict'])
        print("Existing model %s loaded.\n" % (opt.checkpoint))
    device = torch.device("cuda:0")
    ae.to(device)
    ae.eval() # set the autoencoder to evaluation mode

    # neural atlas
    test = neat.loss_metric.test.Test(
        input=neat_config.input,
        num_charts=1,
        metric_config=neat_config.metric,
        uv_space_scale=neat_config.data.uv_space_scale,
        pcl_normalization_scale=neat_config.data.uv_space_scale,
        eval_target_pcl_nml_size=neat_config.data.eval_target_pcl_nml_size,
        model=ae,
        opt=opt
    )

    # Create a folder to write the results
    if not os.path.exists(opt.exp_name):
        os.makedirs(opt.exp_name)

    # Create a tensorboard writer
    if opt.tf_summary: writer = SummaryWriter(log_dir=opt.exp_name)

    # test
    it = 0
    final_metric = easydict.EasyDict({})

    for bi, batch in enumerate(tqdm.tqdm(test_dataloader)):
        batch = easydict.EasyDict({
            "dataset": batch
        })

        # send tensors to GPU
        if opt.svr:
            batch.dataset.input.img = batch.dataset.input.img.to(device)
        else:
            batch.dataset.input.pcl = batch.dataset.input.pcl.to(device)
        batch.dataset.target.pcl = batch.dataset.target.pcl.to(device)
        batch.dataset.target.nml = batch.dataset.target.nml.to(device)
        batch.dataset.target.area = batch.dataset.target.area.to(device)

        mean_metric = test.test_step(batch)

        batch_size = batch.dataset.target.pcl.shape[0]
        for metric_name, mean_metric_value in mean_metric.items():
            if bi == 0:
                final_metric[metric_name] = 0.0
            final_metric[metric_name] += mean_metric_value.item() * batch_size

    for metric_name, final_metric_value in final_metric.items():
        final_metric[metric_name] = final_metric_value / len(test_dataset)
        writer.add_scalar(f'test/{metric_name}', final_metric[metric_name], it)
        print(f'test/{metric_name}: {final_metric[metric_name]}')
        time.sleep(0.01)    # delay a bit to allow the logs to be flushed

    # save the test metrics to the log dir
    METRICS_FILENAME = "metrics.yaml"
    metrics_filepath = os.path.join(opt.exp_name, METRICS_FILENAME)
    with open(metrics_filepath, 'w') as f:
        yaml.dump(dict(final_metric), f)

if __name__ == "__main__":

    option_handler = TestOptionHandler()
    opt = option_handler.parse_options() # all options are parsed through this command
    option_handler.print_options(opt) # print out all the options
    main()
