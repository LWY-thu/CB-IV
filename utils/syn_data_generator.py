import numpy as np
import pandas as pd
import scipy.special
import csv
import sys
import os
from scipy.stats import norm
from scipy.stats import bernoulli    

from utils import * 




class Syn_Generator_LWY(object):   
    def __init__(self, n,ate,sc,sh,one,depX,depU,VX,
                 mV,mX,mU,mXs,init_seed=4,seed_coef=10,details=0,
                 storage_path='./Data/', random_coef='F',use_one='F'):
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

        self.random_coef = random_coef
        self.use_one = use_one

        assert mV<=mX, 'Assume: the dimension of the IVs is less than Confounders'
        
        # coefs_t_VXU 1: Random normal generation; 2: fixed coefficient; 3: 1 coefficient
        np.random.seed(1 * seed_coef * init_seed)
        if random_coef == "True" or random_coef == "T":
            self.coefs_t_VXU = np.random.normal(size=mV + mX + mU)
        else:
            self.coefs_t_VXU = np.round(np.random.uniform(low=8, high=16, size=mV + mX + mU))
        if use_one == "True" or use_one == "T":
            self.coefs_t_VXU = np.ones(shape=mV + mX + mU)
    
        # coefs_y_XU: 1: Random normal generation; 2: fixed coefficient; 3: 1 coefficient
        np.random.seed(2 * seed_coef * init_seed)  # <--
        if random_coef == "True" or random_coef == "T":
            self.coefs_y_XU0 = np.random.normal(size=mX+mU)
            self.coefs_y_XU1 = np.random.normal(size=mX+mU)
        else:
            self.coefs_y_XU0 = np.round(np.random.uniform(low=8, high=16, size=mX + mU))
            self.coefs_y_XU1 = np.round(np.random.uniform(low=8, high=16, size=mX + mU))
        if use_one == "True" or use_one == "T":
            self.coefs_y_XU0 = np.ones(shape=mX+mU)
            self.coefs_y_XU1 = np.ones(shape=mX+mU)
            

        self.set_path(details)
        
        with open(self.data_path+'coefs.csv', 'w') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=',')
            csv_writer.writerow(self.coefs_t_VXU)
            csv_writer.writerow(self.coefs_y_XU0)
            csv_writer.writerow(self.coefs_y_XU1)
        
        # mu, sig = self.get_normal_params(mV=mV, mX=mX+mXs, mU=mU, depX=depX, depU=depU)
        # self.set_normal_params(mu, sig)

        # with open(self.data_path+'norm.csv', 'w') as csvfile:
        #     csv_writer = csv.writer(csvfile, delimiter=',')
        #     csv_writer.writerow(mu)
        #     for row in range(len(sig)):
        #         csv_writer.writerow(sig[row])


    def get_multivariate_normal_params(self, dep, m, seed=0):
        np.random.seed(seed)

        if dep:
            mu = np.zeros(shape=m)
            ''' sample random positive semi-definite matrix for cov '''
            temp = np.random.uniform(size=(m,m))
            temp = .5*(np.transpose(temp)+temp)
            sig = (temp + m*np.eye(m))/10.
        else:
            mu = np.zeros(m)
            sig = np.eye(m)

        return mu, sig

    def get_latent(self, n, m, dep, seed):
        L = np.array((n*[[]]))
        if m != 0:
            mu, sig = self.get_multivariate_normal_params(dep, m, seed)
            L = np.random.multivariate_normal(mean=mu, cov=sig, size=n)
        return L

    def set_path(self,details):
        which_benchmark = 'SynOOD2_'+'_'.join(str(item) for item in [self.sc, self.sh, self.one, self.depX, self.depU,self.VX])
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
        
        seed_coef = self.seed_coef # 种子系数，用于生成随机种子
        init_seed = self.seed # 初始随机种子

        if n is None:
            n = self.n

        print('Next, run dataGenerator: ')

        train_dict, train_df = self.get_data(n=n, seed=seed_coef*init_seed)
        print("1")
        all_df = train_df

        data_path = self.data_path 
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        os.makedirs(os.path.dirname(data_path+'info/'), exist_ok=True)
    
        train_df.to_csv(data_path + '/raw.csv', index=False)

        num_pts = 250
        plot(train_dict['z'][:num_pts], train_dict['pi'][:num_pts], train_dict['t'][:num_pts], train_dict['y'][:num_pts],data_path)

        with open(data_path+'info/specs.csv', 'a') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=',')
            temp = [np.mean(all_df['t'].values), np.min(all_df['pi'].values), np.max(all_df['pi'].values), np.mean(all_df['pi'].values), np.std(all_df['pi'].values)]
            temp.append(lindisc_np(get_var_df(all_df,'x'), all_df['t'].values, np.mean(all_df['t'].values)))
            csv_writer.writerow(temp)
            
        with open(data_path+'info/mu.csv', 'w') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=',')
            E_T_C = ACE(get_var_df(train_df,'m'),train_df['t'].values)
            csv_writer.writerow(E_T_C)

        ''' bias rate '''
        br = [-3.0, -2.5, -2.0, -1.5, -1.3, 1.3, 1.5, 2.0, 2.5, 3.0, 0.0]
        brdc = {-3.0: 'n30', -2.5:'n25', -2.0:'n20', -1.5:'n15', -1.3:'n13', 1.3:'p13', 1.5:'p15', 2.0:'p20', 2.5:'p25', 3.0:'p30', 0.0:'0'}

        for exp in range(num_reps):
            # 在每次迭代中，调用get_data方法生成训练集、验证集和测试集的数据字典和数据框。
            print(f'Run {exp}/{num_reps}. ')
            for r in br:
                train_df_ood = correlation_sample(train_df, r, n, self.mXs)
                val_df_ood = correlation_sample(train_df, r, n, self.mXs)
                test_df_ood = correlation_sample(train_df, r, n, self.mXs)

                path = data_path + '/{}/'.format(exp)
                os.makedirs(os.path.dirname(path + f'ood_{brdc[r]}/'), exist_ok=True)

                train_df_ood.to_csv(path + f'ood_{brdc[r]}/train.csv', index=False)
                val_df_ood.to_csv(path + f'ood_{brdc[r]}/val.csv', index=False)
                test_df_ood.to_csv(path + f'ood_{brdc[r]}/test.csv', index=False)

                
            
        print('-'*30)



    def get_data(self, n, seed):

        mV = self.mV
        mX = self.mX
        mU = self.mU
        mXs = self.mXs
        # Randomly generated coefficients
        random_coef = self.random_coef
        init_seed = self.seed
        seed_coef = self.seed_coef


        # harder datasets
        dep = 0  # overwright; dep=0 generates harder datasets

        # Big Dataset size for sample
        n_trn = n * 100

        # all dimension
        max_dim = mV + mX + mU + mXs

        # Variables
        # 从多元正态分布中生成一个大小为n的随机样本集，其中均值向量为mu，协方差矩阵为sig
        temp = self.get_latent(n=n_trn, m=max_dim, dep=dep, seed=seed_coef * init_seed + 4)
              
        # 通过切片操作，将生成的样本集分成三个部分：V、X和U。V的列数由mV确定，X的列数由mX确定，U的列数由mU确定。
        V = temp[:, 0:mV]
        X = temp[:, mV:mV+mX]
        U = temp[:, mV+mX:mV+mX+mU]
        Xs = temp[:, mV+mX+mU:mV+mX+mU+mXs]
        X_all = np.concatenate((X, Xs), 1)
        VX = np.concatenate((V,X), 1) # VX: variable related T
        Y_vars = np.concatenate([X,U], axis=1) # XU: variable related Y

        # 生成Treatment
        if self.VX:
            T_vars = np.concatenate([V * X[:, 0:mV],X,U], axis=1)
        else:
            T_vars = np.concatenate([V,X,U], axis=1)
        
        # 生成Treatment
        np.random.seed(1*seed)
        z = np.dot(T_vars, self.coefs_t_VXU)
        per = np.random.normal(size=n_trn)
        pi0_t1 = scipy.special.expit( self.sc*(z+self.sh+per) )
        t = bernoulli.rvs(pi0_t1)
        
        # 计算ATE
        coef_devide_2 = 10
        coef_devide_3 = 10
        np.random.seed(2*seed)	  
        if self.random_coef == "True" or self.random_coef == "T" or self.use_one == "True" or self.use_one == "T":               
            mu_0 = np.dot(Y_vars**1, self.coefs_y_XU0) / (mX+mU)
            mu_1 = np.dot(Y_vars**2, self.coefs_y_XU1) / (mX+mU) + self.ate
        else:
            mu_0 = np.dot(Y_vars**1, self.coefs_y_XU0) / (mX+mU) / coef_devide_2
            mu_1 = np.dot(Y_vars**2, self.coefs_y_XU1) / ((mX+mU) + self.ate)/ coef_devide_3

        # 生成y
        y = np.zeros((n_trn, 2))
        y[:,0] = mu_0 + np.random.normal(loc=0., scale=.01, size=n_trn)
        y[:,1] = mu_1 + np.random.normal(loc=0., scale=.01, size=n_trn)
        print("y")

        yf = np.zeros(n_trn)
        ycf = np.zeros(n_trn)
        # for i, t_i in enumerate(t):
        #     yf = np.append(yf, y[i, int(t_i)])
        #     ycf = np.append(ycf, y[i, int(1-t_i)])

        print("dataframe")
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