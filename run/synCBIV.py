'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-11-03 10:09:47
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-11-03 14:08:08
FilePath: /wyliu/code/CB-IV/run/synCBIV.py
Description: 

Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
'''
from sys import path
path.append(r"../")

import os
import argparse
import pandas as pd
import numpy as np
import torch

from utils import log, CausalDataset
from module.SynCBIV import run as run_SynCBIV

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    # About run setting !!!!
    argparser.add_argument('--seed',default=2021,type=int,help='The random seed')
    argparser.add_argument('--mode',default='vx',type=str,help='The choice of v/x/vx/xx')
    argparser.add_argument('--rewrite_log',default=False,type=bool,help='Whether rewrite log file')
    argparser.add_argument('--use_gpu',default=False,type=bool,help='The use of GPU')
    # About data setting ~~~~
    argparser.add_argument('--num',default=10000,type=int,help='The num of train\val\test dataset')
    argparser.add_argument('--num_reps',default=10,type=int,help='The num of train\val\test dataset')
    argparser.add_argument('--ate',default=0,type=float,help='The ate of constant')
    argparser.add_argument('--sc',default=1,type=float,help='The sc')
    argparser.add_argument('--sh',default=0,type=float,help='The sh')
    argparser.add_argument('--one',default=1,type=int,help='The dim of Instrumental variables V')
    argparser.add_argument('--depX',default=0.05,type=float,help='Whether generates harder datasets')
    argparser.add_argument('--depU',default=0.05,type=float,help='Whether generates harder datasets')
    argparser.add_argument('--VX',default=1,type=int,help='The dim of Instrumental variables V')
    argparser.add_argument('--mV',default=2,type=int,help='The dim of Instrumental variables V')
    argparser.add_argument('--mX',default=4,type=int,help='The dim of Confounding variables X')
    argparser.add_argument('--mU',default=4,type=int,help='The dim of Unobserved confounding variables U')
    argparser.add_argument('--storage_path',default='../Data/',type=str,help='The dir of data storage')
    # Syn
    argparser.add_argument('--syn_alpha',default=0.01,type=float,help='')
    argparser.add_argument('--syn_lambda',default=0.0001,type=float,help='')
    argparser.add_argument('--syn_twoStage',default=True,type=bool,help='')
    # About Debug or Show
    argparser.add_argument('--verbose',default=1,type=int,help='The level of verbose')
    argparser.add_argument('--epoch_show',default=5,type=int,help='The epochs of show time')
    args = argparser.parse_args(args=[])
    return args

args = get_args()

if args.use_gpu:
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else "cpu")
else:
    device = torch.device('cpu')
    
# set path
which_benchmark = 'Syn_'+'_'.join(str(item) for item in [args.sc, args.sh, args.one, args.depX, args.depU,args.VX])
which_dataset = '_'.join(str(item) for item in [args.mV, args.mX, args.mU])
resultDir = args.storage_path + f'/results/{which_benchmark}_{which_dataset}/'
dataDir = f'{args.storage_path}/data/{which_benchmark}/{which_dataset}/'
os.makedirs(os.path.dirname(resultDir), exist_ok=True)
logfile = f'{resultDir}/log.txt'

if args.rewrite_log:
    f = open(logfile,'w')
    f.close()

results = []
alpha = args.syn_alpha
for exp in range(args.num_reps):
    # load data
    train_df = pd.read_csv(dataDir + f'{exp}/train.csv')
    val_df = pd.read_csv(dataDir + f'{exp}/val.csv')
    test_df = pd.read_csv(dataDir + f'{exp}/test.csv')
    # train_df = pd.read_csv(dataDir + f'{exp}/{args.mode}/train.csv')
    # val_df = pd.read_csv(dataDir + f'{exp}/{args.mode}/val.csv')
    # test_df = pd.read_csv(dataDir + f'{exp}/{args.mode}/test.csv')

    train = CausalDataset(train_df, variables = ['u','x','v','z','p','s','m','t','g','y','f','c'])
    val = CausalDataset(val_df, variables = ['u','x','v','z','p','s','m','t','g','y','f','c'])
    test = CausalDataset(test_df, variables = ['u','x','v','z','p','s','m','t','g','y','f','c'])

    res_list = []
    
    args.syn_twoStage = False
    args.syn_alpha = 0
    mse_val, obj_val, final = run_SynCBIV(exp, args, dataDir, resultDir, train, val, test, device)
    res_list = res_list + [obj_val['ate_train'],obj_val['ate_test']]
    
    args.syn_twoStage = False
    args.syn_alpha = alpha
    mse_val, obj_val, final = run_SynCBIV(exp, args, dataDir, resultDir, train, val, test, device)
    res_list = res_list + [obj_val['ate_train'],obj_val['ate_test']]
    
    args.syn_twoStage = True
    args.syn_alpha = 0
    mse_val, obj_val, final = run_SynCBIV(exp, args, dataDir, resultDir, train, val, test, device)
    res_list = res_list + [obj_val['ate_train'],obj_val['ate_test']]
    
    args.syn_twoStage = True
    args.syn_alpha = alpha
    mse_val, obj_val, final = run_SynCBIV(exp, args, dataDir, resultDir, train, val, test, device)
    res_list = res_list + [obj_val['ate_train'],obj_val['ate_test']]
    
    res = np.array(res_list) - 1.0
    results.append(res)

results.append(np.mean(results,0))
results.append(np.std(results,0))
    
res_df = pd.DataFrame(np.array(results),
                       columns=[ alpha+data_cls for alpha in ['Direct', 'CFR', 'TwoStage', 'CBIV'] for data_cls in ['_train', '_test']]).round(4)
res_df.to_csv(resultDir + f'CBIV_{args.mode}_result.csv', index=False)
