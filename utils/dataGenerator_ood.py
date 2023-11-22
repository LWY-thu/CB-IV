import numpy as np
import pandas as pd
import scipy.special
import csv
import sys
import os
from scipy.stats import norm

from utils import * 

class Syn_Generator_OOD(object):   
    def __init__(self, n,ate,sc,sh,one,depX,depU,VX,mV,mX,mU,mXs,init_seed=7,seed_coef=10,details=0,storage_path='./Data/'):
        self.n = n # 数据集总量
        self.ate = ate
        self.sc = sc
        self.sh = sh
        self.depX = depX
        self.depU = depU
        self.one = one
        self.VX = VX
        self.mV = mV # IV维数
        self.mX = mX # 可观测到的confounder维数,invariant feature
        self.mU = mU # unmeasured confounder 维数
        self.mXs = mXs # spurious feature
        self.seed = init_seed
        self.seed_coef = seed_coef
        self.storage_path = storage_path

        assert mV<=mX, 'Assume: the dimension of the IVs is less than Confounders'
        
        if one: # 如果参数one为True，则系数被设置为全1；
            self.coefs_VXU = np.ones(shape=mV+mX+mU)
            self.coefs_XU0 = np.ones(shape=mX+mU)
            self.coefs_XU1 = np.ones(shape=mX+mU)
        else: # 否则，系数会从正态分布中随机生成。
            np.random.seed(1*seed_coef*init_seed+3)	          # <--
            self.coefs_VXU = np.random.normal(size=mV+mX+mU)
            
            np.random.seed(2*seed_coef*init_seed+5)	# <--
            self.coefs_XU0 = np.random.normal(size=mX+mU)
            self.coefs_XU1 = np.random.normal(size=mX+mU)
            

        self.set_path(details)
        
        with open(self.data_path+'coefs.csv', 'w') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=',')
            csv_writer.writerow(self.coefs_VXU)
            csv_writer.writerow(self.coefs_XU0)
            csv_writer.writerow(self.coefs_XU1)
        
        mu, sig = self.get_normal_params(mV, mX, mU, depX, depU)
        self.set_normal_params(mu, sig)

        with open(self.data_path+'norm.csv', 'w') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=',')
            csv_writer.writerow(mu)
            for row in range(len(sig)):
                csv_writer.writerow(sig[row])

    def get_normal_params(self, mV, mX, mU, depX, depU):
        m = mV + mX + mU
        mu = np.zeros(m)
        
        sig = np.eye(m)
        temp_sig = np.ones(shape=(m-mV,m-mV))
        temp_sig = temp_sig * depU
        sig[mV:,mV:] = temp_sig

        sig_temp = np.ones(shape=(mX,mX)) * depX
        sig[mV:-mU,mV:-mU] = sig_temp

        sig[np.diag_indices_from(sig)] = 1

        return mu, sig

    def set_normal_params(self, mu, sig):
        self.mu = mu
        self.sig = sig
            
    def set_path(self,details):
        which_benchmark = 'SynOOD_'+'_'.join(str(item) for item in [self.sc, self.sh, self.one, self.depX, self.depU,self.VX])
        print(which_benchmark)
        data_path = self.storage_path+'/data/'+which_benchmark
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        which_dataset = '_'.join(str(item) for item in [self.mV, self.mX, self.mU, self.mXs])
        data_path += '/'+which_dataset+'/'
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        self.data_path = data_path
        self.which_benchmark = which_benchmark
        self.which_dataset = which_dataset

        if details:
            print('#'*30)
            print('The data path is: {}'.format(self.data_path))
            print('The ATE:')
            print('-'*30)
            print(f'ate: {1+self.ate}')  
            print('-'*30)
        
    def run(self, n=None, num_reps=10):
        self.num_reps = num_reps
        
        mu = self.mu  # 均值向量，表示多元正态分布的均值
        sig = self.sig # 协方差矩阵，表示多元正态分布的协方差
        seed_coef = self.seed_coef # 种子系数，用于生成随机种子
        init_seed = self.seed # 初始随机种子

        if n is None:
            n = self.n

        print('Next, run dataGenerator: ')

        for perm in range(num_reps):
            # 在每次迭代中，调用get_data方法生成训练集、验证集和测试集的数据字典和数据框。
            print(f'Run {perm}/{num_reps}. ')
            train_dict, train_df = self.get_data(n, mu, sig, 3*seed_coef*init_seed+perm+777)
            val_dict, val_df = self.get_data(n, mu, sig, 4*seed_coef*init_seed+perm+777)
            test_dict, test_df = self.get_data(n, mu, sig, 5*seed_coef*init_seed+perm+777)
            all_df = train_df.append([val_df, test_df])

            data_path = self.data_path + '/{}/'.format(perm)
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            os.makedirs(os.path.dirname(data_path+'info/'), exist_ok=True)
        
            train_df.to_csv(data_path + '/train.csv', index=False)
            val_df.to_csv(data_path + '/val.csv', index=False)
            test_df.to_csv(data_path + '/test.csv', index=False)

            num_pts = 250
            plot(train_dict['z'][:num_pts], train_dict['pi'][:num_pts], train_dict['t'][:num_pts], train_dict['y'][:num_pts],data_path)

            with open(data_path+'info/specs.csv'.format(perm), 'a') as csvfile:
                csv_writer = csv.writer(csvfile, delimiter=',')
                temp = [np.mean(all_df['t'].values), np.min(all_df['pi'].values), np.max(all_df['pi'].values), np.mean(all_df['pi'].values), np.std(all_df['pi'].values)]
                temp.append(lindisc_np(get_var_df(all_df,'x'), all_df['t'].values, np.mean(all_df['t'].values)))
                csv_writer.writerow(temp)
                
            with open(data_path+'info/mu.csv'.format(perm), 'w') as csvfile:
                csv_writer = csv.writer(csvfile, delimiter=',')
                E_T_C = ACE(get_var_df(train_df,'m'),train_df['t'].values)
                csv_writer.writerow(E_T_C)
                E_T_C = ACE(get_var_df(val_df,'m'),val_df['t'].values)
                csv_writer.writerow(E_T_C)
                E_T_C = ACE(get_var_df(val_df,'m'),val_df['t'].values)
                csv_writer.writerow(E_T_C)

        print('-'*30)
            
    def get_data(self, n, mu, sig, seed):
        np.random.seed(seed)

        mV = self.mV
        mX = self.mX
        mU = self.mU
        mXs = self.mXs
        
        # 从多元正态分布中生成一个大小为n的随机样本集，其中均值向量为mu，协方差矩阵为sig
        temp = np.random.multivariate_normal(mean=mu, cov=sig, size=n)
        # 通过切片操作，将生成的样本集分成三个部分：V、X和U。V的列数由mV确定，X的列数由mX确定，U的列数由mU确定。
        V = temp[:, 0:mV]
        X = temp[:, mV:mV+mX]
        U = temp[:, mV+mX:mV+mX+mU]
        Xs = temp[:, mV+mX+mXs:mV+mX+mU+mXs]

        if self.VX:
            T_vars = np.concatenate([V * X[:, 0:mV],X,U], axis=1)
        else:
            T_vars = np.concatenate([V,X,U], axis=1)
        Y_vars = np.concatenate([X,U], axis=1)
        
        # 生成Treatment
        np.random.seed(2*seed)	                # <--------------
        z = np.dot(T_vars, self.coefs_VXU)
        pi0_t1 = scipy.special.expit( self.sc*(z+self.sh) )
        t = np.array([])
        for p in pi0_t1:
            t = np.append(t, np.random.binomial(1, p, 1))

        # 计算ATE    
        mu_0 = np.dot(Y_vars**1, self.coefs_XU0) / (mX+mU)
        mu_1 = np.dot(Y_vars**2, self.coefs_XU1) / (mX+mU) + self.ate
        # 生成y
        np.random.seed(3*seed)	                # <--------------
        y = np.zeros((n, 2))
        y[:,0] = mu_0 + np.random.normal(loc=0., scale=.01, size=n)
        y[:,1] = mu_1 + np.random.normal(loc=0., scale=.01, size=n)

        yf = np.array([])
        ycf = np.array([])
        for i, t_i in enumerate(t):
            yf = np.append(yf, y[i, int(t_i)])
            ycf = np.append(ycf, y[i, int(1-t_i)])
        
        # V:工具变量 X:Observed confounder U:Unmeasured confounder Xs:spurious feature
        # z, pi: pi=P(T|Z,X)=1/1+exp(z) t:Treatment 
        # mu0:对照组的平均因果响应（Average Treatment Effect for Control Group），表示对照组在未接受处理时的平均因果影响。它是一个常数。
        # mu1:处理组的平均因果响应（Average Treatment Effect for Treated Group），表示处理组在接受处理时的平均因果影响。它是一个常数。
        #yf:Factual Outcome ycf:Counterfactual Outcome y:实际观测到的结果变量
        data_dict = {'V':V, 'U':U, 'X':X, 'Xs':Xs,'z':z, 'pi':pi0_t1, 't':t, 'mu0':mu_0, 'mu1':mu_1, 'yf':yf, 'y':y, 'ycf':ycf}
        data_all = np.concatenate([V, X, U, Xs, z.reshape(-1,1), pi0_t1.reshape(-1,1), t.reshape(-1,1), mu_0.reshape(-1,1), mu_1.reshape(-1,1), yf.reshape(-1,1), ycf.reshape(-1,1)], axis=1)
        data_df = pd.DataFrame(data_all,
                               columns=['v{}'.format(i+1) for i in range(V.shape[1])] + 
                               ['x{}'.format(i+1) for i in range(X.shape[1])] + 
                               ['u{}'.format(i+1) for i in range(U.shape[1])] + 
                               ['xs{}'.format(i+1) for i in range(Xs.shape[1])] + 
                               ['z','pi','t','mu0','mu1','y','f'])
        
        return data_dict, data_df
