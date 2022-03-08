'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''

import os
from os.path import join
import torch
from parse import parse_args
import multiprocessing
import yaml


yaml.warnings({'YAMLLoadWarning': False})


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
args = parse_args()

ROOT_PATH = ""
CODE_PATH = join(ROOT_PATH, 'code')
DATA_PATH = join(ROOT_PATH, 'data')
BOARD_PATH = join(CODE_PATH, 'runs')
FILE_PATH = join(CODE_PATH, 'checkpoints')
import sys
sys.path.append(join(CODE_PATH, 'sources'))


if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH, exist_ok=True)

with open('./config.yml') as f:
        params = yaml.safe_load(f)

config = {}
all_dataset = ['lastfm', 'gowalla', 'yelp2018', 'amazon-book','own','my_data']
all_models  = ['mf', 'lgn']
# config['batch_size'] = 4096
config['bpr_batch_size'] = params['bpr_batch']
config['wandb'] = params['wandb']
config['wandb_name'] = params['wandb_name']
config['latent_dim_rec'] = params['recdim']
config['lightGCN_n_layers']= params['layer']
config['dropout'] = params['dropout']
config['keep_prob']  = params['keep_prob']
config['A_n_fold'] = params['A_n_fold']
config['test_u_batch_size'] =  params['test_u_batch_size']
config['multicore'] = params['multicore']
config['lr'] = params['lr']
config['decay'] = params['decay']
config['pretrain'] = params['pretrain']
config['A_split'] = False
config['bigdata'] = False

GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else "cpu")
CORES = params['cores']
seed = params['seed']

dataset = params['dataset']
model_name = params['model']
if dataset not in all_dataset:
    raise NotImplementedError(f"Haven't supported {dataset} yet!, try {all_dataset}")
if model_name not in all_models:
    raise NotImplementedError(f"Haven't supported {model_name} yet!, try {all_models}")




TRAIN_epochs = params['epochs']
LOAD = params['load']
PATH = params['path']
topks = eval(params['topks'])
tensorboard = params['tensorboard']
comment = params['comment']
# let pandas shut up
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)



def cprint(words : str):
    print(f"\033[0;30;43m{words}\033[0m")

logo = r"""
██╗      ██████╗ ███╗   ██╗
██║     ██╔════╝ ████╗  ██║
██║     ██║  ███╗██╔██╗ ██║
██║     ██║   ██║██║╚██╗██║
███████╗╚██████╔╝██║ ╚████║
╚══════╝ ╚═════╝ ╚═╝  ╚═══╝
"""
# font: ANSI Shadow
# refer to http://patorjk.com/software/taag/#p=display&f=ANSI%20Shadow&t=Sampling
# print(logo)
