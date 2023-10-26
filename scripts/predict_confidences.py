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
import yaml

@click.command()
### Add your options here
@click.option(
    "--config",
    "-c",
    type=str,
    default="./config/predict_config_wayside.yaml",
    help="config file.",
)
def main(config):
    cfg=yaml.safe_load(open(config))
    print(cfg)
    data = datasets.KittiSequentialModule(cfg)
    data.setup()

    ckpt = torch.load(cfg['MODEL']['WEIGHT'])
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
