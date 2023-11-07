from sys import path
path.append(r"../")

dataDir='../Data/Causal/'
storage_path='../Data/'

from utils import Syn_Generator, IHDP_Generator, Twins_Generator

Syn_244 = Syn_Generator(n=10000, ate=0,sc=1,sh=0,one=1,depX=0.05,depU=0.05,VX=1,mV=2,mX=4,mU=4,init_seed=7,seed_coef=10,details=1,storage_path=storage_path)
Syn_244.run(n=10000, num_reps=10)