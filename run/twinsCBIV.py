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
import time
import os
import argparse
import pandas as pd
import numpy as np
import torch
sys.path.append(r"../")
sys.path.append(r"../../")
sys.path.append('/home/wyliu/code/CB-IV')
from utils import log, CausalDataset
from module.SynCBIV import run as run_SynCBIV
from module.TwinsCBIV import run as run_TwinsCBIV

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def run(args):   
    if args.use_gpu:
        device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else "cpu")
    else:
        device = torch.device('cpu')
    #set path
    des_str = args.des_str
    which_benchmark = 'Twins_'+'_'.join(str(item) for item in [args.sc, args.sh, args.one, args.VX])
    which_dataset = '_'.join(str(item) for item in [args.mV, args.mX, args.mU])
    resultDir = args.storage_path + f'/results/{which_benchmark}_{which_dataset}_{args.mode}/'
    dataDir = f'{args.storage_path}/data/{which_benchmark}/{which_dataset}/'
    os.makedirs(os.path.dirname(resultDir), exist_ok=True)
    resultDir = resultDir + des_str
    os.makedirs(os.path.dirname(resultDir), exist_ok=True)
    logfile = f'{resultDir}/log.txt'
        

    if args.rewrite_log:
        f = open(logfile,'w')
        f.close()

    results_ate = []
    results_pehe = []
    results_loss = []
    alpha = args.syn_alpha
    for exp in range(args.num_reps):
        # load data
        # train_df = pd.read_csv(dataDir + f'{exp}/train.csv')
        # val_df = pd.read_csv(dataDir + f'{exp}/val.csv')
        # test_df = pd.read_csv(dataDir + f'{exp}/test.csv')
        train_df = pd.read_csv(dataDir + f'{exp}/{args.mode}/train.csv')
        val_df = pd.read_csv(dataDir + f'{exp}/{args.mode}/val.csv')
        test_df = pd.read_csv(dataDir + f'{exp}/{args.mode}/test.csv')

        train = CausalDataset(train_df, variables = ['u','x','v','z','p','s','m','t','g','y','f','c'], observe_vars=['v', 'x'])
        val = CausalDataset(val_df, variables = ['u','x','v','z','p','s','m','t','g','y','f','c'], observe_vars=['v', 'x'])
        test = CausalDataset(test_df, variables = ['u','x','v','z','p','s','m','t','g','y','f','c'], observe_vars=['v', 'x'])

        res_ate_list = []
        res_pehe_list = []
        res_loss_list = []
        
        # args.syn_twoStage = False
        # args.syn_alpha = 0
        # mse_val, obj_val, final = run_SynCBIV(exp, args, dataDir, resultDir, train, val, test, device)
        # res_ate_list = res_ate_list + [obj_val['ate_train'],obj_val['ate_test']]
        # res_pehe_list = res_pehe_list + [obj_val['pehe_train'],obj_val['pehe_test']]
        
        # args.syn_twoStage = False
        # args.syn_alpha = alpha
        # mse_val, obj_val, final = run_SynCBIV(exp, args, dataDir, resultDir, train, val, test, device)
        # res_ate_list = res_ate_list + [obj_val['ate_train'],obj_val['ate_test']]
        # res_pehe_list = res_pehe_list + [obj_val['pehe_train'],obj_val['pehe_test']]
        
        # args.syn_twoStage = True
        # args.syn_alpha = 0
        # mse_val, obj_val, final = run_SynCBIV(exp, args, dataDir, resultDir, train, val, test, device)
        # res_ate_list = res_ate_list + [obj_val['ate_train'],obj_val['ate_test']]
        # res_pehe_list = res_pehe_list + [obj_val['pehe_train'],obj_val['pehe_test']]
        
        args.syn_twoStage = True
        args.syn_alpha = alpha
        start_time = time.time()
        train_obj_val, train_f_val, valid_obj_val, valid_f_val, final = run_TwinsCBIV(exp, args, dataDir, resultDir, train, val, test, device)
        end_time = time.time()
        run_time = end_time - start_time
        logfile = f'{resultDir}/log.txt'
        _logfile = f'{resultDir}/CBIV.txt'
        log(logfile, f'End: exp:{exp}; run_time:{run_time};')
        log(_logfile, f'End: exp:{exp}; lrate:{run_time};',False)
        res_ate_list = res_ate_list + [train_obj_val['ate_train'],train_obj_val['ate_test'], 
                                       train_f_val['ate_train'],train_f_val['ate_test'], 
                                       valid_obj_val['ate_train'],valid_obj_val['ate_test'],
                                       valid_f_val['ate_train'],valid_f_val['ate_test'], 
                                       ]
        res_pehe_list = res_pehe_list + [train_obj_val['pehe_train'],train_obj_val['pehe_test'], 
                                         train_f_val['pehe_train'],train_f_val['pehe_test'],
                                         valid_obj_val['pehe_train'],valid_obj_val['pehe_test'],
                                         valid_f_val['pehe_train'],valid_f_val['pehe_test'], 
                                         ]
        res_loss_list = res_loss_list + [train_obj_val['best'],
                                       train_f_val['best'],
                                       valid_obj_val['best'],
                                       valid_f_val['best'],
                                       ]
        
        # res = np.array(res_ate_list) - 1.0
        # res_pehe = np.array(res_pehe_list) - 1.0
        results_ate.append(res_ate_list)
        results_pehe.append(res_pehe_list)
        results_loss.append(res_loss_list)

    results_ate.append(np.mean(results_ate[:][:args.num_reps],0))
    results_ate.append(np.std(results_ate[:][:args.num_reps],0))
    results_pehe.append(np.mean(results_pehe[:][:args.num_reps],0))
    results_pehe.append(np.std(results_pehe[:][:args.num_reps],0))
    results_loss.append(np.mean(results_loss[:][:args.num_reps],0))
    results_loss.append(np.std(results_loss[:][:args.num_reps],0))
        
    res_ate_df = pd.DataFrame(np.array(results_ate),
                        columns=[ alpha+data_cls for alpha in ['tr_obj', 'tr_f', ' val_obf', ' val_f'] for data_cls in ['_tr', '_te']]).round(4)
    res_ate_df.to_csv(resultDir + f'CBIV_{args.mode}_ate_result.csv', index=False)
    results_pehe = pd.DataFrame(np.array(results_pehe),
                        columns=[ alpha+data_cls for alpha in ['tr_obj', 'tr_f', ' val_obf', ' val_f'] for data_cls in ['_tr', '_te']]).round(4)
    results_pehe.to_csv(resultDir + f'CBIV_{args.mode}_pehe_result.csv', index=False)
    res_loss_df = pd.DataFrame(np.array(results_loss),
                        columns=[ 'train_obj', 'train_f', ' valid_obf', ' valid_f']).round(4)
    res_loss_df.to_csv(resultDir + f'CBIV_{args.mode}_loss.csv', index=False)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    # About run setting !!!!
    argparser.add_argument('--seed',default=2021,type=int,help='The random seed')
    argparser.add_argument('--mode',default='vx',type=str,help='The choice of v/x/vx/xx')
    argparser.add_argument('--rewrite_log',default=False,type=bool,help='Whether rewrite log file')
    argparser.add_argument('--use_gpu',default=True,type=bool,help='The use of GPU')
    argparser.add_argument('--des_str',default='/_/',type=str,help='The description of this running')
    argparser.add_argument('--iter',default=3000,type=int,help='The num of iterations')
    # About data setting ~~~~
    argparser.add_argument('--num',default=3321,type=int,help='The num of train\val\test dataset')
    argparser.add_argument('--num_reps',default=10,type=int,help='The num of train\val\test dataset')
    argparser.add_argument('--ate',default=-0.0252,type=float,help='The ate of constant')
    argparser.add_argument('--sc',default=1,type=float,help='The sc')
    argparser.add_argument('--sh',default=-2,type=float,help='The sh')
    argparser.add_argument('--one',default=1,type=int,help='The dim of Instrumental variables V')
    argparser.add_argument('--VX',default=1,type=int,help='The dim of Instrumental variables V')
    argparser.add_argument('--mV',default=5,type=int,help='The dim of Instrumental variables V')
    argparser.add_argument('--mX',default=5,type=int,help='The dim of Confounding variables X')
    argparser.add_argument('--mU',default=3,type=int,help='The dim of Unobserved confounding variables U')
    argparser.add_argument('--storage_path',default='../../Data/',type=str,help='The dir of data storage')
    # Twins
    argparser.add_argument('--twins_alpha',default=0.0001,type=float,help='')
    argparser.add_argument('--twins_lambda',default=0.0001,type=float,help='')
    argparser.add_argument('--twins_twoStage',default=True,type=bool,help='')
    # Syn
    argparser.add_argument('--syn_alpha',default=0.0001,type=float,help='')
    argparser.add_argument('--syn_lambda',default=0.0001,type=float,help='')
    argparser.add_argument('--syn_twoStage',default=True,type=bool,help='')
    argparser.add_argument('--lrate',default=0.001,type=float,help='learning rate')

    # About Debug or Show
    argparser.add_argument('--verbose',default=1,type=int,help='The level of verbose')
    argparser.add_argument('--epoch_show',default=5,type=int,help='The epochs of show time')
    args = argparser.parse_args()

    run(args=args)
