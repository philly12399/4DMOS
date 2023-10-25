#!/usr/bin/env python3
# @file      predict_confidences.py
# @author    Benedikt Mersch     [mersch@igg.uni-bonn.de]
# Copyright (c) 2022 Benedikt Mersch, all rights reserved

import click
from pytorch_lightning import Trainer
import torch
import torch.nn.functional as F
import os
import mos4d.datasets.datasets as datasets
import mos4d.models.models as models


@click.command()
### Add your options here
@click.option(
    "--weights",
    "-w",
    type=str,
    help="path to checkpoint file (.ckpt) to do inference.",
    required=True,
)
@click.option(
    "--sequence",
    "-seq",
    type=int,
    help="Run inference on a specific sequence. Otherwise, test split from config is used.",
    default=None,
    multiple=True,
)
@click.option(
    "--dt",
    "-dt",
    type=float,
    help="Desired temporal resolution of predictions.",
    default=None,
)
@click.option("--poses", "-poses", type=str, default=None, help="Specify which poses to use.")
@click.option(
    "--transform",
    "-transform",
    type=bool,
    default=None,
    help="Transform point clouds to common viewpoint.",
)
@click.option(
    "--config",
    "-c",
    type=str,
    default="./config/semantic-kitti-mos.yaml",
    help="config file.",
)
def main(weights, sequence, dt, poses, transform, config):

    cfg = torch.load(weights)["hyper_parameters"]
    # print(config)
    
    if poses:
        cfg["DATA"]["POSES"] = poses

    if transform != None:
        cfg["DATA"]["TRANSFORM"] = transform
        if not transform:
            cfg["DATA"]["POSES"] = "no_poses"
    if sequence:
        cfg["DATA"]["SPLIT"]["TEST"] = list(sequence)

    if dt:
        cfg["MODEL"]["DELTA_T_PREDICTION"] = dt
    cfg["TRAIN"]["BATCH_SIZE"] = 1
   
    # Load data and model
    cfg["DATA"]["SPLIT"]["TRAIN"] = cfg["DATA"]["SPLIT"]["TEST"]
    cfg["DATA"]["SPLIT"]["VAL"] = cfg["DATA"]["SPLIT"]["TEST"]
    #CUstomize
    cfg["DATA"]["TRANSFORM"] = False
    cfg["DATA"]["POSES"] = "no_poses"
    cfg["DATA"]["NUM_WORKER"]=8
    cfg["DATA"]["SPLIT"]["TRAIN"] = []
    cfg["DATA"]["SPLIT"]["VAL"] = []
    cfg["DATA"]["SPLIT"]["TEST"]=[0]
    print(cfg)
    data = datasets.KittiSequentialModule(cfg)
    data.setup()

    ckpt = torch.load(weights)
    model = models.MOSNet(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model = model.cuda()
    model.eval()
    model.freeze()

    # Setup trainer
    trainer = Trainer(accelerator='gpu', devices=1)

    # Infer!
    trainer.predict(model, data.test_dataloader())


if __name__ == "__main__":
    main()
