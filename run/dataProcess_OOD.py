'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-11-03 10:09:47
LastEditors: lwy_thu 760835659@qq.com
LastEditTime: 2023-11-06 12:48:02
FilePath: /wyliu/code/CB-IV/run/synCBIV.py
Description: 

Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
'''
import sys

import os
import argparse
import pandas as pd
import numpy as np
import torch

sys.path.append(r"../")
sys.path.append('/home/wyliu/code/CB-IV')
from utils import * 
from utils import log, CausalDataset, Syn_Generator_OOD
# from module.SynCBIV import run as run_SynCBIV
from module.SynCBIV_OOD import run as run_SynCBIV
from module.Regression_OOD import run as run_Reg

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def run(args):
    Syn_2442 = Syn_Generator_OOD(n=args.ood_num, 
                                 ate=args.ate,
                                 sc=args.sc,
                                 sh=args.sh,
                                 one=args.one,
                                 depX=args.depX,
                                 depU=args.depU,
                                 VX=args.VX,
                                 mV=args.mV,
                                 mX=args.mX,
                                 mU=args.mU,
                                 mXs=args.mXs,
                                 init_seed=7,
                                 seed_coef=10,
                                 details=1,
                                 storage_path=args.storage_path)
    Syn_2442.run(n=args.ood_num, num_reps=args.num_reps)

    Datasets = [Syn_2442]

    # run vx
    for mode in ['x', 'vx']:
        data = Datasets[0]
        which_benchmark = data.which_benchmark
        which_dataset = data.which_dataset
        args.num_reps = 10
        args.mV = data.mV
        args.mX = data.mX
        args.mU = data.mU
        args.mD = data.mXs
        args.mode = mode

        resultDir = args.storage_path + f'/results/{which_benchmark}_{which_dataset}/'
        dataDir = f'{args.storage_path}/data/{which_benchmark}/{which_dataset}/'
        os.makedirs(os.path.dirname(resultDir), exist_ok=True)
        logfile = f'{resultDir}/log.txt'

        if args.rewrite_log:
            f = open(logfile,'w')
            f.close()

        for exp in range(args.num_reps):
            train_df = pd.read_csv(dataDir + f'{exp}/train.csv')
            print(dataDir)
            val_df = pd.read_csv(dataDir + f'{exp}/val.csv')
            test_df = pd.read_csv(dataDir + f'{exp}/test.csv')
                                                        
            train = CausalDataset(train_df, variables = ['v','u','x','xs','z','p','s','m','t','g','y','f','c'])
            val = CausalDataset(val_df, variables = ['v','u','x','xs','z','p','s','m','t','g','y','f','c'])
            test = CausalDataset(test_df, variables = ['v','u','x','xs','z','p','s','m','t','g','y','f','c'])

            train,val,test = run_Reg(exp, args, dataDir, resultDir, train, val, test, device)
    
        ''' bias rate '''
        br = [-3.0, -2.5, -2.0, -1.5, -1.3, 1.3, 1.5, 2.0, 2.5, 3.0, 0.0]
        brdc = {-3.0: 'n30', -2.5:'n25', -2.0:'n20', -1.5:'n15', -1.3:'n13', 1.3:'p13', 1.5:'p15', 2.0:'p20', 2.5:'p25', 3.0:'p30', 0.0:'0'}
        
        # set path
        which_benchmark = 'SynOOD_'+'_'.join(str(item) for item in [args.sc, args.sh, args.one, args.depX, args.depU,args.VX])
        which_dataset = '_'.join(str(item) for item in [args.mV, args.mX, args.mU, args.mD])
        resultDir = args.storage_path + f'/results/{which_benchmark}_{which_dataset}/'
        dataDir = f'{args.storage_path}/data/{which_benchmark}/{which_dataset}/'
        os.makedirs(os.path.dirname(resultDir), exist_ok=True)
        logfile = f'{resultDir}/log.txt'
        exp = 0
        
        for perm in range(args.num_reps):
            for r in br:
                train_df = pd.read_csv(dataDir + f'{perm}/{args.mode}/train.csv')
                val_df = pd.read_csv(dataDir + f'{perm}/{args.mode}/val.csv')
                test_df = pd.read_csv(dataDir + f'{perm}/{args.mode}/test.csv')

                train_df_ood = correlation_sample(train_df, r, args.num, args.mXs)
                val_df_ood = correlation_sample(val_df, r, args.num, args.mXs)
                test_df_ood = correlation_sample(test_df, r, args.num, args.mXs)

                path = dataDir + '/{}/{}/'.format(perm, args.mode)
                os.makedirs(os.path.dirname(path + f'ood_{brdc[r]}/'), exist_ok=True)

                train_df_ood.to_csv(path + f'ood_{brdc[r]}/train.csv', index=False)
                val_df_ood.to_csv(path + f'ood_{brdc[r]}/val.csv', index=False)
                test_df_ood.to_csv(path + f'ood_{brdc[r]}/test.csv', index=False)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    # About run setting !!!!
    argparser.add_argument('--seed',default=2021,type=int,help='The random seed')
    argparser.add_argument('--mode',default='vx',type=str,help='The choice of v/x/vx/xx')
    argparser.add_argument('--ood',default=0,type=float,help='The train dataset of OOD')
    argparser.add_argument('--rewrite_log',default=False,type=bool,help='Whether rewrite log file')
    argparser.add_argument('--use_gpu',default=True,type=bool,help='The use of GPU')
    # About data setting ~~~~
    argparser.add_argument('--ood_num',default=1000000,type=int,help='The num of train\val\test dataset')
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
    argparser.add_argument('--mXs',default=2,type=int,help='The dim of Noise variables X')
    argparser.add_argument('--storage_path',default='../Data/',type=str,help='The dir of data storage')
    # Syn
    argparser.add_argument('--syn_alpha',default=0.01,type=float,help='')
    argparser.add_argument('--syn_lambda',default=0.0001,type=float,help='')
    argparser.add_argument('--syn_twoStage',default=True,type=bool,help='')
    # About Debug or Show
    argparser.add_argument('--verbose',default=1,type=int,help='The level of verbose')
    argparser.add_argument('--epoch_show',default=5,type=int,help='The epochs of show time')
    # About Regression_t
    argparser.add_argument('--regt_batch_size',default=500,type=int,help='The size of one batch')
    argparser.add_argument('--regt_lr',default=0.05,type=float,help='The learning rate')
    argparser.add_argument('--regt_num_epoch',default=3,type=int,help='The num of total epoch')
    args = argparser.parse_args()

    if args.use_gpu:
        device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else "cpu")
    run(args=args)