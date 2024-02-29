'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-11-01 12:30:07
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-11-03 07:56:48
FilePath: /wyliu/code/CB-IV/module/Regression.py
Description: 这段代码实现了一个多层感知机MLP模型以及相关的训练和评估过程。

Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
'''
import os
import random
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils import set_seed, log

def get_gain(activation):
    if activation.__class__.__name__ == "LeakyReLU":
        gain = nn.init.calculate_gain("leaky_relu",
                                            activation.negative_slope)
    else:
        activation_name = activation.__class__.__name__.lower()
        try:
            gain = nn.init.calculate_gain(activation_name)
        except ValueError:
            gain = 1.0
    return gain

# input_dim：输入数据的维度。
# layer_widths：一个整数列表，表示隐藏层的宽度。
# activation：激活函数（默认为 None）。
# last_layer：可选的最后一层，可以是任何 nn.Module 的子类（默认为 None）。
# num_out：输出的维度（默认为 1）。
class MLPModel(nn.Module):
    def __init__(self, input_dim, layer_widths, activation=None,last_layer=None, num_out=1):
        nn.Module.__init__(self)
        self.gain=get_gain(activation)
        # 根据隐藏层的宽度列表 layer_widths，
        # 创建一系列的线性层（nn.Linear），
        # 并可选择地在每个线性层之后添加给定的激活函数 activation。
        # 最后，根据输出维度 num_out 添加最后一层线性层。
        if len(layer_widths) == 0:
            layers = [nn.Linear(input_dim, num_out)]
        else:
            num_layers = len(layer_widths)
            if activation is None:
                layers = [nn.Linear(input_dim, layer_widths[0])]
            else:
                layers = [nn.Linear(input_dim, layer_widths[0]), activation]
            for i in range(1, num_layers):
                w_in = layer_widths[i-1]
                w_out = layer_widths[i]
                if activation is None:
                    layers.extend([nn.Linear(w_in, w_out)])
                else:
                    layers.extend([nn.Linear(w_in, w_out), activation])
            layers.append(nn.Linear(layer_widths[-1], num_out))
        if last_layer:
            layers.append(last_layer)
        self.model = nn.Sequential(*layers)

    def initialize(self, gain=1.0):
        # initialize 方法用于初始化模型的参数。
        for layer in self.model[:-1]:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight.data, gain=self.gain)
                nn.init.zeros_(layer.bias.data)
        final_layer = self.model[-1]
        nn.init.xavier_normal_(final_layer.weight.data, gain=gain)
        nn.init.zeros_(final_layer.bias.data)

    def forward(self, data):
        # print("forward", data.shape)
        num_data = data.shape[0]
        data = data.view(num_data, -1)
        return self.model(data)

class MultipleMLPModel(nn.Module):
    def __init__(self, input_dim, layer_widths, num_models=1, activation=None,last_layer=None, num_out=1):
        nn.Module.__init__(self)
        self.models = nn.ModuleList([MLPModel(
            input_dim, layer_widths, activation=activation,
            last_layer=last_layer, num_out=num_out) for _ in range(num_models)])
        self.num_models = num_models

    def forward(self, data):
        num_data = data.shape[0]
        data = data.view(num_data, -1)
        outputs = [self.models[i](data) for i in range(self.num_models)]
        return torch.cat(outputs, dim=1)

def run(exp, args, dataDir, resultDir, train, val, test, device, r):
    batch_size = args.regt_batch_size
    lr = args.regt_lr
    num_epoch = args.regt_num_epoch
    logfile = f'{resultDir}/log.txt'
    _logfile = f'{resultDir}/Regression.txt'
    set_seed(args.seed)

    try:
        train.to_tensor()
        val.to_tensor()
        test.to_tensor()
    except:
        pass

    train_loader = DataLoader(train, batch_size=batch_size)
    if args.mode == 'v':
        input_dim = args.mV
        train_input = train.v
        val_input = val.v
        test_input = test.v
    elif args.mode == 'x':
        input_dim = args.mX + args.mXs
        train_input = torch.cat((train.x, train.xs),1)
        val_input = torch.cat((val.x, val.xs),1)
        test_input = torch.cat((test.x, test.xs),1)
    else:
        input_dim = args.mV + args.mX + args.mXs
        # print("input dim:", input_dim)
        train_input = torch.cat((train.v, train.x, train.xs),1)
        val_input = torch.cat((val.v, val.x, val.xs),1)
        test_input = torch.cat((test.v, test.x, test.xs),1)

    
    mlp = MLPModel(input_dim, layer_widths=[128, 64], activation=nn.ReLU(),last_layer=nn.BatchNorm1d(2), num_out=2)
    net = nn.Sequential(mlp)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss_func = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epoch):
        log(logfile, f"Exp {exp} :this is the {epoch}/{num_epoch} epochs.")
        log(_logfile, f"Exp {exp} :this is the {epoch}/{num_epoch} epochs.", False)
        for idx, inputs in enumerate(train_loader):
            u = inputs['u']
            v = inputs['v']
            x = torch.cat((inputs['x'], inputs['xs']), 1)
            t = inputs['t'].reshape(-1).type(torch.LongTensor)
            # print("x:", x.shape)
            # print("args.mode:",args.mode)
            if args.mode == 'v':
                input_batch = v
            elif args.mode == 'x':
                input_batch = x
                # print("input_batch:", input_batch.shape)
            else:
                input_batch = torch.cat((v, x),1)
            
            prediction = net(input_batch) 
            loss = loss_func(prediction, t)

            optimizer.zero_grad()  
            loss.backward()        
            optimizer.step()    

        log(logfile, 'The train accuracy: {:.2f} %'.format((torch.true_divide(sum(train.t.reshape(-1) == torch.max(F.softmax(net(train_input) , dim=1), 1)[1]), len(train.t))).item() * 100))
        log(_logfile, 'The test  accuracy: {:.2f} %'.format((torch.true_divide(sum(test.t.reshape(-1) == torch.max(F.softmax(net(test_input) , dim=1), 1)[1]), len(test.t))).item() * 100))

    train.s = F.softmax(net(train_input) , dim=1)[:,1:2]
    val.s = F.softmax(net(val_input) , dim=1)[:,1:2]
    test.s = F.softmax(net(test_input) , dim=1)[:,1:2]
    ''' bias rate 1'''
    br = [-3.0, -2.5, -2.0, -1.5, -1.3, 1.3, 1.5, 2.0, 2.5, 3.0]
    brdc = {-3.0: 'n30', -2.5:'n25', -2.0:'n20', -1.5:'n15', -1.3:'n13', 1.3:'p13', 1.5:'p15', 2.0:'p20', 2.5:'p25', 3.0:'p30', 0.0:'0'}

    os.makedirs(os.path.dirname(dataDir + f'{exp}/ood_{brdc[r]}/{args.mode}/'), exist_ok=True)

    train.to_cpu()
    train.detach()
    tmp_df = train.to_pandas()
    tmp_df.to_csv(dataDir + f'{exp}/ood_{brdc[r]}/{args.mode}/train.csv', index=False)

    val.to_cpu()
    val.detach()
    tmp_df = val.to_pandas()
    tmp_df.to_csv(dataDir + f'{exp}/ood_{brdc[r]}/{args.mode}/val.csv', index=False)

    test.to_cpu()
    test.detach()
    tmp_df = test.to_pandas()
    tmp_df.to_csv(dataDir + f'{exp}/ood_{brdc[r]}/{args.mode}/test.csv', index=False)

    return train,val,test

# 在第一阶段加入OOD                 
def run_ood_stage1(exp, args, dataDir, resultDir, train, val, test, test_ood, device, ood_test_dict):
    batch_size = args.regt_batch_size
    lr = args.regt_lr
    num_epoch = args.regt_num_epoch
    logfile = f'{resultDir}/log.txt'
    _logfile = f'{resultDir}/Regression.txt'
    set_seed(args.seed)

    try:
        train.to_tensor()
        val.to_tensor()
        test.to_tensor()
        test_ood.to_tensor()
    except:
        pass

    train_loader = DataLoader(train, batch_size=batch_size)
    if args.mode == 'v':
        input_dim = args.mV
        train_input = train.v
        val_input = val.v
        test_input = test.v
    elif args.mode == 'x':
        input_dim = args.mX + args.mXs
        train_input = torch.cat((train.x, train.xs),1)
        val_input = torch.cat((val.x, val.xs),1)
        test_input = torch.cat((test.x, test.xs),1)
    else:
        input_dim = args.mV + args.mX + args.mXs
        # print("input dim:", input_dim)
        train_input = torch.cat((train.v, train.x, train.xs),1)
        val_input = torch.cat((val.v, val.x, val.xs),1)
        test_input = torch.cat((test.v, test.x, test.xs),1)
        test_ood_input = torch.cat((test_ood.v, test_ood.x, test_ood.xs),1)

    
    mlp = MLPModel(input_dim, layer_widths=[128, 64], activation=nn.ReLU(),last_layer=nn.BatchNorm1d(2), num_out=2)
    net = nn.Sequential(mlp)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss_func = torch.nn.CrossEntropyLoss()

    log(_logfile, f"Exp: {exp}; regt_lr:{lr}")
    for epoch in range(num_epoch):
        log(logfile, f"Exp {exp} :this is the {epoch}/{num_epoch} epochs.")
        log(_logfile, f"Exp {exp} :this is the {epoch}/{num_epoch} epochs.", False)
        for idx, inputs in enumerate(train_loader):
            u = inputs['u']
            v = inputs['v']
            x = torch.cat((inputs['x'], inputs['xs']), 1)
            t = inputs['t'].reshape(-1).type(torch.LongTensor)
            # print("x:", x.shape)
            # print("args.mode:",args.mode)
            if args.mode == 'v':
                input_batch = v
            elif args.mode == 'x':
                input_batch = x
                # print("input_batch:", input_batch.shape)
            else:
                input_batch = torch.cat((v, x),1)
            
            prediction = net(input_batch) 
            loss = loss_func(prediction, t)

            optimizer.zero_grad()  
            loss.backward()        
            optimizer.step()    

        log(logfile, 'The train accuracy: {:.2f} %'.format((torch.true_divide(sum(train.t.reshape(-1) == torch.max(F.softmax(net(train_input) , dim=1), 1)[1]), len(train.t))).item() * 100))
        log(_logfile, 'The train accuracy: {:.2f} %'.format((torch.true_divide(sum(train.t.reshape(-1) == torch.max(F.softmax(net(train_input) , dim=1), 1)[1]), len(train.t))).item() * 100))
        log(_logfile, 'The test  accuracy: {:.2f} %'.format((torch.true_divide(sum(test.t.reshape(-1) == torch.max(F.softmax(net(test_input) , dim=1), 1)[1]), len(test.t))).item() * 100))
        log(_logfile, 'The test_ood  accuracy: {:.2f} %'.format((torch.true_divide(sum(test_ood.t.reshape(-1) == torch.max(F.softmax(net(test_ood_input) , dim=1), 1)[1]), len(test_ood.t))).item() * 100))

    train.s = F.softmax(net(train_input) , dim=1)[:,1:2]
    val.s = F.softmax(net(val_input) , dim=1)[:,1:2]
    test.s = F.softmax(net(test_input) , dim=1)[:,1:2]
    ''' bias rate 1'''
    br = [-3.0, -2.5, -2.0, -1.5, -1.3, 1.3, 1.5, 2.0, 2.5, 3.0]
    brdc = {-3.0: 'n30', -2.5:'n25', -2.0:'n20', -1.5:'n15', -1.3:'n13', 1.3:'p13', 1.5:'p15', 2.0:'p20', 2.5:'p25', 3.0:'p30', 0.0:'0'}
    for r in br:
        test_ood = ood_test_dict[r]
        try:
            test_ood.to_tensor()
        except:
            pass
        if args.mode == 'v':
            input_dim = args.mV
            test_input_ood = test_ood.v
        elif args.mode == 'x':
            input_dim = args.mX + args.mXs
            test_input_ood = torch.cat((test_ood.x, test_ood.xs),1)
        else:
            input_dim = args.mV + args.mX + args.mXs
            test_input_ood = torch.cat((test_ood.v, test_ood.x, test_ood.xs),1)
        test_ood.s = F.softmax(net(test_input_ood) , dim=1)[:,1:2]
        test_ood.to_cpu()
        test_ood.detach()
        ood_test_dict[r] = test_ood

    train.to_cpu()
    train.detach()

    val.to_cpu()
    val.detach()

    test.to_cpu()
    test.detach()

    return train, val, test, ood_test_dict