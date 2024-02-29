import numpy as np
from scipy.stats import norm
import matplotlib
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from sklearn.model_selection import train_test_split

FLAGS = tf.app.flags.FLAGS

def get_multivariate_normal_params(dep, m, seed=0):
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

def get_latent(n, m, dep, seed):
    L = np.array((n*[[]]))
    if m != 0:
        mu, sig = get_multivariate_normal_params(dep, m, seed)
        L = np.random.multivariate_normal(mean=mu, cov=sig, size=n)
    return L

def ACE(mu,t):
    mu0 = mu[:,0:1]
    mu1 = mu[:,1:2]
        
    it = np.where(t>0.5)
    ic = np.where(t<0.5)

    mu0_t = mu0[it]
    mu1_t = mu1[it]
    
    mu0_c = mu0[ic]
    mu1_c = mu1[ic]

    return [np.mean(mu0),np.mean(mu1),np.mean(mu1)-np.mean(mu0),np.mean(mu0_t),np.mean(mu1_t),np.mean(mu1_t)-np.mean(mu0_t),np.mean(mu0_c),np.mean(mu1_c),np.mean(mu1_c)-np.mean(mu0_c)]
    
def lindisc_np(X,t,p):
    ''' Linear MMD '''

    it = np.where(t>0)
    ic = np.where(t<1)

    Xc = X[ic]
    Xt = X[it]

    mean_control = np.mean(Xc,axis=0)
    mean_treated = np.mean(Xt,axis=0)

    c = np.square(2*p-1)*0.25
    f = np.sign(p-0.5)

    mmd = np.sum(np.square(p*mean_treated - (1-p)*mean_control))
    mmd = f*(p-0.5) + np.sqrt(c + mmd)

    return mmd

def plot(z, pi0_t1, t, y, data_path):
    gridspec.GridSpec(3,1)

    z_min = np.min(z) #- np.std(z)
    z_max = np.max(z) #+ np.std(z)
    z_grid = np.linspace(z_min, z_max, 100)

    ax = plt.subplot2grid((3,1), (0,0), rowspan=2)
    ind = np.where(t==0)
    plt.plot(z[ind], np.squeeze(y[ind,0]), '+', color='r')
    ind = np.where(t==1)
    plt.plot(z[ind], np.squeeze(y[ind,1]), '.', color='b')
    plt.legend(['t=0', 't=1'])

    ax = plt.subplot2grid((3,1), (2,0), rowspan=1)
    ind = np.where(t==0)
    mu, std = norm.fit(z[ind])
    p = norm.pdf(z_grid, mu, std)
    plt.plot(z_grid, p, color='r', linewidth=2)
    ind = np.where(t==1)
    mu, std = norm.fit(z[ind])
    p = norm.pdf(z_grid, mu, std)
    plt.plot(z_grid, p, color='b', linewidth=2)

    plt.savefig(data_path+'info/distribution.png')
    plt.close()
    
def get_var_df(df,var):
    var_cols = [c for c in df.columns if c.startswith(var)]
    return df[var_cols].to_numpy()


'''
description: 
param {*} data 输入数据
param {*} r bias rate 
param {*} n size
param {*} dim_v Dimensions of irrelevant variables
return {*}
'''
def correlation_sample(data, r, n, dim_xs):
    nall = data.shape[0]
    prob = np.ones(nall)

    ite = data['mu1']-data['mu0']

    if r!=0.0:
        for idv in range(dim_xs):
            # 和表达式不一样
            d = np.abs(data[f'xs{(idv+1)}'] - np.sign(r) * ite) 
            prob = prob * np.power(np.abs(r), -10 * d)
    prob = prob / np.sum(prob)
    idx = np.random.choice(range(nall), n, p=prob, replace=False)

    new_data = data.iloc[idx].reset_index(drop=True)
    t = new_data['t']
    mu0 = new_data['mu0']
    mu1 = new_data['mu1']

    # # continuous y
    # y0_cont = mu0 + np.random.normal(loc=0., scale=.1, size=n)
    # y1_cont = mu1 + np.random.normal(loc=0., scale=.1, size=n)

    # yf_cont, ycf_cont = pd.Series(np.zeros(n), dtype=float), pd.Series(np.zeros(n), dtype=float)
    # yf_cont[t>0], yf_cont[t<1] = y1_cont[t>0], y0_cont[t<1]
    # ycf_cont[t>0], ycf_cont[t<1] = y0_cont[t>0], y1_cont[t<1]

    # new_data['mu0'] = y0_cont
    # new_data['mu1'] = y1_cont
    # new_data['y'] = yf_cont
    # new_data['f'] = ycf_cont

    # binary y
    median_0 = np.median(mu0)
    median_1 = np.median(mu1)
    mu0[mu0 >= median_0] = 1.
    mu0[mu0 < median_0] = 0.
    mu1[mu1 < median_1] = 0.
    mu1[mu1 >= median_1] = 1.

    yf_bin, ycf_bin = np.zeros(n), np.zeros(n)
    yf_bin[t>0], yf_bin[t<1] = mu1[t>0], mu0[t<1]
    ycf_bin[t>0], ycf_bin[t<1] = mu0[t>0], mu1[t<1]

    new_data['mu0'] = mu0
    new_data['mu1'] = mu1
    new_data['y'] = yf_bin
    new_data['f'] = ycf_bin

    return new_data

def pehe(ypred1, ypred0, mu1, mu0):
    return np.sqrt(np.mean(np.square((mu1 - mu0) - (ypred1 - ypred0))))

def split_data(data):
    # 划分训练集和剩余数据
    train_data, remaining_data = train_test_split(data, test_size=0.3, shuffle=True, random_state=42)

    # 划分测试集和验证集
    test_data, val_data = train_test_split(remaining_data, test_size=(1/3), shuffle=True, random_state=42)

    # 打印划分后的数据集大小
    print("训练集大小:", len(train_data))
    print("测试集大小:", len(test_data))
    print("验证集大小:", len(val_data))

    # 进行后续操作，使用划分后的数据集
    return train_data, val_data, test_data

def save_config(fname):
    """ Save configuration """
    flagdict =  {}
    for k in FLAGS:
        flagdict[k] = FLAGS[k].value
    s = '\n'.join(['%s: %s' % (k,str(flagdict[k])) for k in sorted(flagdict.keys())])
    f = open(fname,'w')
    f.write(s)
    f.close()