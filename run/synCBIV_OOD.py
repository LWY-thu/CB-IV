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
sys.path.append(r"../../")
sys.path.append('/home/wyliu/code/CB-IV')
from utils import log, CausalDataset
from utils import *
import time
# from module.SynCBIV import run as run_SynCBIV
# from module.SynCBIV_OODv1 import run as run_SynCBIV


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def run(args):   
    if args.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else "cpu")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
        device = torch.device('cpu')
    print('device:', device)
    if args.version == 0:
        print("version0")
        from module.SynCBIV_OODv0 import run as run_SynCBIV
    elif args.version == 1:
        print("version1")
        from module.SynCBIV_OODv1 import run as run_SynCBIV
    # set path
    ''' bias rate 1'''
    br = [-3.0, -2.5, -2.0, -1.5, -1.3, 1.3, 1.5, 2.0, 2.5, 3.0, 0.0]
    brdc = {-3.0: 'n30', -2.5:'n25', -2.0:'n20', -1.5:'n15', -1.3:'n13', 1.3:'p13', 1.5:'p15', 2.0:'p20', 2.5:'p25', 3.0:'p30', 0.0:'0'}

    des_str = args.des_str
    which_benchmark = 'SynOOD2_'+'_'.join(str(item) for item in [args.sc, args.sh, args.one, args.depX, args.depU,args.VX])
    which_dataset = '_'.join(str(item) for item in [args.mV, args.mX, args.mU, args.mXs])
    resultDir = args.storage_path + f'/results/{which_benchmark}_{which_dataset}_{args.mode}/ood{brdc[args.ood]}/'
    os.makedirs(os.path.dirname(resultDir), exist_ok=True)
    resultDir = resultDir + des_str
    os.makedirs(os.path.dirname(resultDir), exist_ok=True)
    dataDir = f'{args.storage_path}/data/{which_benchmark}/{which_dataset}/'
    
    logfile = f'{resultDir}/log.txt'

    if args.rewrite_log:
        f = open(logfile,'w')
        f.close()

    results_ate = []
    results_pehe = []
    results_loss = []
    results_ood_ate_direct = []
    results_ood_pehe_direct = []
    results_ood_ate_cfr = []
    results_ood_pehe_cfr = []
    results_ood_ate_twostage = []
    results_ood_pehe_twostage = []
    # cbiv模型下四种loss对应的ood test结果
    results_ood_ate_tr_obj = []
    results_ood_ate_tr_f = []
    results_ood_ate_val_obj = []
    results_ood_ate_val_f = []
    results_ood_pehe_tr_obj = []
    results_ood_pehe_tr_f = []
    results_ood_pehe_val_obj = []
    results_ood_pehe_val_f = []
    # results_ood_earlystop = []
    # results_ood = [results_ood_ate_direct, results_ood_pehe_direct,
    #                results_ood_ate_cfr, results_ood_pehe_cfr,
    #                results_ood_ate_twostage, results_ood_pehe_twostage,
    #                results_ood_ate_cbiv, results_ood_pehe_cbiv]
    # name_ood = ["results_ood_ate_direct", "results_ood_pehe_direct",
    #                "results_ood_ate_cfr", "results_ood_pehe_cfr",
    #                "results_ood_ate_twostage", "results_ood_pehe_twostage",
    #                "results_ood_ate_cbiv", "results_ood_pehe_cbiv"]
    results_ood = [results_ood_ate_tr_obj, results_ood_ate_tr_f,
                   results_ood_ate_val_obj, results_ood_ate_val_f,
                   results_ood_pehe_tr_obj, results_ood_pehe_tr_f,
                   results_ood_pehe_val_obj, results_ood_pehe_val_f]
    
    name_ood = ['results_ood_ate_tr_obj', 'results_ood_ate_tr_f',
                   'results_ood_ate_val_obj', 'results_ood_ate_val_f',
                   'results_ood_pehe_tr_obj', 'results_ood_pehe_tr_f',
                   'results_ood_pehe_val_obj', 'results_ood_pehe_val_f']
    alpha = args.syn_alpha
    for exp in range(args.num_reps):
        # # load data v0
        train_df = pd.read_csv(dataDir + f'{exp}/ood_{brdc[args.ood]}/{args.mode}/train.csv')
        # val_df = pd.read_csv(dataDir + f'{exp}/ood_{brdc[args.ood]}/{args.mode}/val.csv')
        # test_df = pd.read_csv(dataDir + f'{exp}/ood_{brdc[args.ood]}/{args.mode}/test.csv')

        # load data v1
        # train_df = pd.read_csv(dataDir + f'{exp}/ood_{brdc[args.ood]}/train.csv')

        train_df, val_df, test_df = split_data(train_df)

        train = CausalDataset(train_df, variables = ['u','x','v','xs','z','p','s','m','t','g','y','f','c'], observe_vars=['v','x','xs'])
        val = CausalDataset(val_df, variables = ['u','x','v','xs','z','p','s','m','t','g','y','f','c'], observe_vars=['v','x','xs'])
        test = CausalDataset(test_df, variables = ['u','x','v','xs','z','p','s','m','t','g','y','f','c'], observe_vars=['v','x','xs'])

        res_ate_list = []
        res_pehe_list = []
        res_loss_list = []
                
        args.syn_twoStage = False
        args.syn_alpha = alpha
        start_time = time.time()
        train_obj_val, train_f_val, valid_obj_val, valid_f_val, final= run_SynCBIV(exp, args, dataDir, resultDir, train, val, test, device)
        end_time = time.time()
        run_time = end_time - start_time
        logfile = f'{resultDir}/log.txt'
        _logfile = f'{resultDir}/CBIV.txt'
        log(logfile, f'End: exp:{exp}; run_time:{run_time};')
        log(_logfile, f'End: exp:{exp}; run_time:{run_time};',False)


        res_ate_list = res_ate_list + [train_obj_val['ate_train'],train_obj_val['ate_test'], train_obj_val['ate_ood'],
                                       train_f_val['ate_train'],train_f_val['ate_test'], train_f_val['ate_ood'],
                                       valid_obj_val['ate_train'],valid_obj_val['ate_test'], valid_obj_val['ate_ood'],
                                       valid_f_val['ate_train'],valid_f_val['ate_test'], valid_f_val['ate_ood']
                                       ]
        res_pehe_list = res_pehe_list + [train_obj_val['pehe_train'],train_obj_val['pehe_test'], train_obj_val['pehe_ood'],
                                         train_f_val['pehe_train'],train_f_val['pehe_test'], train_f_val['pehe_ood'],
                                         valid_obj_val['pehe_train'],valid_obj_val['pehe_test'], valid_obj_val['pehe_ood'],
                                         valid_f_val['pehe_train'],valid_f_val['pehe_test'], valid_f_val['pehe_ood']
                                         ]
        res_loss_list = res_loss_list + [train_obj_val['best'],
                                       train_f_val['best'],
                                       valid_obj_val['best'],
                                       valid_f_val['best'],
                                       ]
        if args.oodtestall ==1 :
            results_ood_ate_tr_obj.append(train_obj_val['ate_ood_list'])
            results_ood_ate_tr_f.append(train_f_val['ate_ood_list'])
            results_ood_ate_val_obj.append(valid_obj_val['ate_ood_list'])
            results_ood_ate_val_f.append(valid_f_val['ate_ood_list'])
            results_ood_pehe_tr_obj.append(train_obj_val['pehe_ood_list'])
            results_ood_pehe_tr_f.append(train_f_val['pehe_ood_list'])
            results_ood_pehe_val_obj.append(valid_obj_val['pehe_ood_list'])
            results_ood_pehe_val_f.append(valid_f_val['pehe_ood_list'])

        
        # res = np.array(res_ate_list) - 1.0
        # res_pehe = np.array(res_pehe_list) - 1.0
        # results_ood_earlystop.append(ood_earlystop_temp)
        results_ate.append(res_ate_list)
        results_pehe.append(res_pehe_list)
        results_loss.append(res_loss_list)

    ''' bias rate 1'''
    br = [-3.0, -2.5, -2.0, -1.5, -1.3, 0.0, 1.3, 1.5, 2.0, 2.5, 3.0]
    brdc = {-3.0: 'n30', -2.5:'n25', -2.0:'n20', -1.5:'n15', -1.3:'n13', 1.3:'p13', 1.5:'p15', 2.0:'p20', 2.5:'p25', 3.0:'p30', 0.0:'0'}
 
    results_ate.append(np.mean(results_ate[:][:args.num_reps],0))
    results_ate.append(np.std(results_ate[:][:args.num_reps],0))
    results_pehe.append(np.mean(results_pehe[:][:args.num_reps],0))
    results_pehe.append(np.std(results_pehe[:][:args.num_reps],0))
    results_loss.append(np.mean(results_loss[:][:args.num_reps],0))
    results_loss.append(np.std(results_loss[:][:args.num_reps],0))
    for res, name in zip(results_ood, name_ood):
        res.append(np.mean(res[:][:args.num_reps],0))
        res.append(np.std(res[:][:args.num_reps],0))
        res_df = pd.DataFrame(np.array(res), columns=[brdc[r] for r in br ]).round(4)
        res_df.to_csv(resultDir + f'CBIV_{args.mode}_' + name + '.csv', index=False)

    
    res_ate_df = pd.DataFrame(np.array(results_ate),
                        columns=[ alpha+data_cls for alpha in ['tr_obj', 'tr_f', ' val_obf', ' val_f'] for data_cls in ['_tr', '_te', '_ood']]).round(4)
    res_ate_df.to_csv(resultDir + f'CBIV_{args.mode}_ate_earlyresult.csv', index=False)
    results_pehe = pd.DataFrame(np.array(results_pehe),
                        columns=[ alpha+data_cls for alpha in ['tr_obj', 'tr_f', ' val_obf', ' val_f'] for data_cls in ['_tr', '_te', '_ood']]).round(4)
    results_pehe.to_csv(resultDir + f'CBIV_{args.mode}_pehe_earlyresult.csv', index=False)
    res_loss_df = pd.DataFrame(np.array(results_loss),
                        columns=[ 'train_obj', 'train_f', ' valid_obf', ' valid_f']).round(4)
    res_loss_df.to_csv(resultDir + f'CBIV_{args.mode}_loss.csv', index=False)

    
    print(f"---------------------ood_{brdc[args.ood]}_end---------------------------")
    

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    # About run setting !!!!
    argparser.add_argument('--seed',default=2021,type=int,help='The random seed')
    argparser.add_argument('--mode',default='vx',type=str,help='The choice of v/x/vx/xx')
    argparser.add_argument('--ood',default=3.0,type=float,help='The train dataset of OOD')
    argparser.add_argument('--ood_test',default=-3.0,type=float,help='The train dataset of OOD')
    argparser.add_argument('--rewrite_log',default=False,type=bool,help='Whether rewrite log file')
    argparser.add_argument('--use_gpu',default=1,type=int,help='The use of GPU')
    argparser.add_argument('--des_str',default='/_/',type=str,help='The description of this running')
    argparser.add_argument('--oodtestall',default=0,type=int,help='The random seed')
    argparser.add_argument('--iter',default=3000,type=int,help='The num of iterations')
    argparser.add_argument('--version',default=1,type=int,help='The version')
    # About data setting ~~~~
    argparser.add_argument('--num',default=10000,type=int,help='The num of train\val\test dataset')
    argparser.add_argument('--num_reps',default=100,type=int,help='The num of train\val\test dataset')
    argparser.add_argument('--ate',default=0,type=float,help='The ate of constant')
    argparser.add_argument('--sc',default=1,type=float,help='The sc')
    argparser.add_argument('--sh',default=0,type=float,help='The sh')
    argparser.add_argument('--one',default=1,type=int,help='The dim of Instrumental variables V')
    argparser.add_argument('--depX',default=0.05,type=float,help='Whether generates harder datasets')
    argparser.add_argument('--depU',default=0.05,type=float,help='Whether generates harder datasets')
    argparser.add_argument('--VX',default=1,type=int,help='The dim of Instrumental variables V')
    argparser.add_argument('--mV',default=2,type=int,help='The dim of Instrumental variables V')
    argparser.add_argument('--mX',default=10,type=int,help='The dim of Confounding variables X')
    argparser.add_argument('--mU',default=4,type=int,help='The dim of Unobserved confounding variables U')
    argparser.add_argument('--mXs',default=2,type=int,help='The dim of Noise variables X')
    argparser.add_argument('--storage_path',default='../../Data/',type=str,help='The dir of data storage')
    # Syn
    argparser.add_argument('--syn_alpha',default=0.01,type=float,help='')
    argparser.add_argument('--syn_lambda',default=0.001,type=float,help='')
    argparser.add_argument('--syn_twoStage',default=True,type=bool,help='')
    argparser.add_argument('--lrate',default=0.001,type=float,help='learning rate')
    # About Debug or Show
    argparser.add_argument('--verbose',default=1,type=int,help='The level of verbose')
    argparser.add_argument('--epoch_show',default=5,type=int,help='The epochs of show time')
    # About Regression_t
    argparser.add_argument('--regt_batch_size',default=500,type=int,help='The size of one batch')
    argparser.add_argument('--regt_lr',default=0.05,type=float,help='The learning rate')
    argparser.add_argument('--regt_num_epoch',default=5,type=int,help='The num of total epoch')
    args = argparser.parse_args()
    
    run(args=args)

