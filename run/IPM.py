import sys
import os
sys.path.append(r"../")
sys.path.append(r"../../")
sys.path.append('/home/wyliu/code/CB-IV')
from utils.imbFun import *
import random
import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from utils import log, CausalDataset
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def mmd2_rbf_ood(Xc,Xt,p,sig):
    """ Computes the l2-RBF MMD for X given t """

    # it = tf.where(t>0)[:,0]
    # ic = tf.where(t<1)[:,0]

    # Xc = tf.gather(X,ic)
    # Xt = tf.gather(X,it)
    Xc = tf.convert_to_tensor(Xc, dtype=tf.float32)
    Xt = tf.convert_to_tensor(Xt, dtype=tf.float32)
    Kcc = tf.exp(-pdist2sq(Xc,Xc)/tf.square(sig))
    Kct = tf.exp(-pdist2sq(Xc,Xt)/tf.square(sig))
    Ktt = tf.exp(-pdist2sq(Xt,Xt)/tf.square(sig))

    m = tf.to_float(tf.shape(Xc)[0])
    n = tf.to_float(tf.shape(Xt)[0])

    mmd = tf.square(1.0-p)/(m*(m-1.0))*(tf.reduce_sum(Kcc)-m)
    mmd = mmd + tf.square(p)/(n*(n-1.0))*(tf.reduce_sum(Ktt)-n)
    mmd = mmd - 2.0*p*(1.0-p)/(m*n)*tf.reduce_sum(Kct)
    mmd = 4.0*mmd

    return mmd

def wasserstein_ood(Xc,Xt, p,lam=10,its=10,sq=False,backpropT=False):
    """ Returns the Wasserstein distance between treatment groups """

    # it = tf.where(t1>0)[:,0]
    # ic = tf.where(t2>0)[:,0]
    # Xc = tf.gather(X1,ic)
    # Xt = tf.gather(X2,it)
    Xc = tf.convert_to_tensor(Xc, dtype=tf.float32)
    Xt = tf.convert_to_tensor(Xt, dtype=tf.float32)
    nc = tf.to_float(tf.shape(Xc)[0])
    nt = tf.to_float(tf.shape(Xt)[0])

    ''' Compute distance matrix'''
    if sq:
        M = pdist2sq(Xt,Xc)
    else:
        M = safe_sqrt(pdist2sq(Xt,Xc))

    ''' Estimate lambda and delta '''
    M_mean = tf.reduce_mean(M)
    M_drop = tf.nn.dropout(M,10/(nc*nt))
    delta = tf.stop_gradient(tf.reduce_max(M))
    eff_lam = tf.stop_gradient(lam/M_mean)

    ''' Compute new distance matrix '''
    Mt = M
    row = delta*tf.ones(tf.shape(M[0:1,:]))
    col = tf.concat([delta*tf.ones(tf.shape(M[:,0:1])),tf.zeros((1,1))],0)
    Mt = tf.concat([M,row],0)
    Mt = tf.concat([Mt,col],1)

    ''' Compute marginal vectors '''
    # print('nt', nt)
    a = tf.concat([p*tf.ones((10000,1))/nt, (1-p)*tf.ones((1,1))],0)
    b = tf.concat([(1-p)*tf.ones((10000,1))/nc, p*tf.ones((1,1))],0)

    ''' Compute kernel matrix'''
    Mlam = eff_lam*Mt
    K = tf.exp(-Mlam) + 1e-6 # added constant to avoid nan
    U = K*Mt
    ainvK = K/a

    u = a
    for i in range(0,its):
        u = 1.0/(tf.matmul(ainvK,(b/tf.transpose(tf.matmul(tf.transpose(u),K)))))
    v = b/(tf.transpose(tf.matmul(tf.transpose(u),K)))

    T = u*(tf.transpose(v)*K)

    if not backpropT:
        T = tf.stop_gradient(T)

    E = T*Mt
    D = 2*tf.reduce_sum(E)

    return D, Mlam


def run(args):
    if args.use_gpu:
        device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else "cpu")
    else:
        device = torch.device('cpu')
    # set OOD path
    ''' bias rate '''
    # args.ood = -3.0
    ''' bias rate 1'''
    br = [-3.0, -2.5, -2.0, -1.5, -1.3, 1.3, 1.5, 2.0, 2.5, 3.0, 0.0]
    brdc = {-3.0: 'n30', -2.5:'n25', -2.0:'n20', -1.5:'n15', -1.3:'n13', 1.3:'p13', 1.5:'p15', 2.0:'p20', 2.5:'p25', 3.0:'p30', 0.0:'0'}
    # ''' bias rate 2'''
    # br = [1.0, 1.3, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    # brdc = {1.0:'p10', 1.3:'p13', 1.5:'p15',2.0:'p20', 2.5:'p25', 3.0:'p30',3.5:'p35', 4.0:'p40',4.5:'p45', 5.0:'p50'}
    which_benchmark = 'SynOOD_'+'_'.join(str(item) for item in [args.sc, args.sh, args.one, args.depX, args.depU,args.VX])
    which_dataset = '_'.join(str(item) for item in [args.mV, args.mX, args.mU, args.mXs])
    resultDir = args.storage_path + f'/results/{which_benchmark}_{which_dataset}_{args.mode}/ood{brdc[args.ood]}/'
    dataDir = f'{args.storage_path}/data/{which_benchmark}/{which_dataset}/'
    os.makedirs(os.path.dirname(resultDir), exist_ok=True)
    logfile = f'{resultDir}/log.txt'
    
    exp=0
    train_df = pd.read_csv(dataDir + f'{exp}/ood_{brdc[args.ood]}/{args.mode}/train.csv')
    val_df = pd.read_csv(dataDir + f'{exp}/ood_{brdc[args.ood]}/{args.mode}/val.csv')
    test_df = pd.read_csv(dataDir + f'{exp}/ood_{brdc[args.ood]}/{args.mode}/test.csv')
    train = CausalDataset(train_df, variables = ['u','x','v','xs','z','p','s','m','t','g','y','f','c'], observe_vars=['v','x','xs'])
    val = CausalDataset(val_df, variables = ['u','x','v','xs','z','p','s','m','t','g','y','f','c'], observe_vars=['v','x','xs'])
    test = CausalDataset(test_df, variables = ['u','x','v','xs','z','p','s','m','t','g','y','f','c'], observe_vars=['v','x','xs'])

    x_list = [np.concatenate((train.x, train.xs), 1), 
                np.concatenate((val.x, val.xs), 1), 
                np.concatenate((test.x, test.xs), 1)]

    train = {'x':x_list[0],
            't':train.t,
            's':train.s,
            'g':train.g,
            'yf':train.y,
            'ycf':train.f}
    val = {'x':x_list[1],
            't':val.t,
            's':val.s,
            'g':val.g,
            'yf':val.y,
            'ycf':val.f}
    test = {'x':x_list[2],
            't':test.t,
            's':test.s,
            'g':test.g,
            'yf':test.y,
            'ycf':test.f}

    wass_dist = []
    mmd_dist = []
    results_ood = [wass_dist, mmd_dist]
    name_ood = ["wass_dist", "mmd_dist"]
    for exp in range(1): 
        l1 = []
        l2 = []
        for r in br:    
            train_df_ood = pd.read_csv(dataDir + f'{exp}/ood_{brdc[r]}/{args.mode}/train.csv')
            # val_df_ood = pd.read_csv(dataDir + f'{exp}/{args.mode}/ood_{brdc[args.ood]}/val.csv')
            # test_df_ood = pd.read_csv(dataDir + f'{exp}/{args.mode}/ood_{brdc[args.ood]}/test.csv')
            train_ood = CausalDataset(train_df_ood, variables = ['u','x','v','xs','z','p','s','m','t','g','y','f','c'], observe_vars=['v','x','xs'])
            # val_ood = CausalDataset(val_df_ood, variables = ['u','x','v','xs','z','p','s','m','t','g','y','f','c'], observe_vars=['v','x','xs'])
            # test_ood = CausalDataset(test_df_ood, variables = ['u','x','v','xs','z','p','s','m','t','g','y','f','c'], observe_vars=['v','x','xs'])
            print(dataDir + f'{exp}/{args.mode}/ood_{brdc[r]}/train.csv')
            x_list = [np.concatenate((train_ood.x, train_ood.xs), 1)]

            train_ood = {'x':x_list[0],
                    't':train_ood.t,
                    's':train_ood.s,
                    'g':train_ood.g,
                    'yf':train_ood.y,
                    'ycf':train_ood.f}
            p_ipm = 0.5
            imb_dist, imB_mat = wasserstein_ood(train['x'],train_ood['x'],p_ipm)
            # 创建 TensorFlow 会话
            with tf.Session() as sess:
                # 执行计算图并获取张量的值
                tensor_value = sess.run(imb_dist)
            print(tensor_value)
            l1.append(tensor_value)

            p_ipm = 0.5
            imb_dist_mmd = mmd2_rbf_ood(train['x'],train_ood['x'],p_ipm, 0.1)
            # 创建 TensorFlow 会话
            with tf.Session() as sess:
                # 执行计算图并获取张量的值
                tensor_value_mmd = sess.run(imb_dist_mmd)
            print(tensor_value_mmd)
            l2.append(tensor_value_mmd)
        wass_dist.append(l1)
        mmd_dist.append(l2)


    for res, name in zip(results_ood, name_ood):
        res.append(np.mean(res[:][:args.num_reps],0))
        res.append(np.std(res[:][:args.num_reps],0))
        if name == "wass_dist":
            res_df = pd.DataFrame(np.array(res), columns=[brdc[r] for r in br ]).round(4)
        else:
            res_df = pd.DataFrame(np.array(res), columns=[brdc[r] for r in br ])
        res_df.to_csv(resultDir + f'IPM_{args.mode}_{brdc[args.ood]}_' + name + '.csv', index=False)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    # About run setting !!!!
    argparser.add_argument('--seed',default=2021,type=int,help='The random seed')
    argparser.add_argument('--mode',default='vx',type=str,help='The choice of v/x/vx/xx')
    argparser.add_argument('--ood',default=0,type=float,help='The train dataset of OOD')
    argparser.add_argument('--rewrite_log',default=False,type=bool,help='Whether rewrite log file')
    argparser.add_argument('--use_gpu',default=True,type=bool,help='The use of GPU')
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
    argparser.add_argument('--mX',default=10,type=int,help='The dim of Confounding variables X')
    argparser.add_argument('--mU',default=4,type=int,help='The dim of Unobserved confounding variables U')
    argparser.add_argument('--mXs',default=2,type=int,help='The dim of Noise variables X')
    argparser.add_argument('--storage_path',default='../../Data/',type=str,help='The dir of data storage')
    # Syn
    argparser.add_argument('--syn_alpha',default=0.01,type=float,help='')
    argparser.add_argument('--syn_lambda',default=0.001,type=float,help='')
    argparser.add_argument('--syn_twoStage',default=True,type=bool,help='')
    # About Debug or Show
    argparser.add_argument('--verbose',default=1,type=int,help='The level of verbose')
    argparser.add_argument('--epoch_show',default=5,type=int,help='The epochs of show time')
    args = argparser.parse_args()

    print(args.mV)
    run(args=args)