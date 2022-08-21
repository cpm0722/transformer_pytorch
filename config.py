"""
@author : Hansu Kim(@cpm0722)
@when : 2022-08-21
@github : https://github.com/cpm0722
@homepage : https://cpm0722.github.io
"""

import torch

DEVICE = torch.device('cuda:0')
CHECKPOINT_DIR = "./checkpoint"

N_EPOCH = 1000

BATCH_SIZE = 2048
NUM_WORKERS = 8

LEARNING_RATE = 1e-5
WEIGHT_DECAY = 5e-4
ADAM_EPS = 5e-9
SCHEDULER_FACTOR = 0.9
SCHEDULER_PATIENCE = 10

WARM_UP_STEP = 100

DROPOUT_RATE = 0.1
