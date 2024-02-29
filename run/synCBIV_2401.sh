# 0106
# python synCBIV_OOD.py --iter=20000 --lrate=0.002 --oodtestall=1 --num_reps=20 --des_str='/0106_lr=0.002_reps=20_v1/' --ood=3.0 --ood_test=-3.0 --ivreg=1&
# python synCBIV_OOD.py --iter=20000 --lrate=0.002 --oodtestall=1 --num_reps=20 --start_reps=20 --des_str='/0106_lr=0.002_reps=20_st=20_v1/' --ood=3.0 --ood_test=-3.0 --ivreg=1&
# python synCBIV_OOD.py --iter=20000 --lrate=0.002 --oodtestall=1 --num_reps=20 --start_reps=40 --des_str='/0106_lr=0.002_reps=20_st=40_v1/' --ood=3.0 --ood_test=-3.0 --ivreg=1&

# 0112
# python synCBIV_OOD.py --iter=8000 --lrate=0.002 --num_reps=2 --des_str='/0112_lr=0.002_dim=256/' --ood=3.0 --ood_test=-3.0 &
# python synCBIV_OOD.py --iter=8000 --lrate=0.002 --num_reps=2 --des_str='/0112_lr=0.002_dim=128/' --ood=3.0 --ood_test=-3.0 &
# python synCBIV_OOD.py --iter=8000 --lrate=0.002 --num_reps=2 --des_str='/0112_lr=0.002_dim=100/' --ood=3.0 --ood_test=-3.0 &
# python synCBIV_OOD.py --iter=8000 --lrate=0.002 --num_reps=2 --des_str='/0112_lr=0.002_dim=384/' --ood=3.0 --ood_test=-3.0 &
# nohup python synCBIV_OOD.py --iter=8000 --lrate=0.002 --num_reps=10 --des_str='/0112_lr=0.002_dim=512/' --ood=3.0 --ood_test=-3.0 &

# 0114
# python synCBIV_OOD.py --iter=8000 --lrate=0.002 --num_reps=10 --des_str='/0114_lr=0.002_iter=8000_dim=100/' --dim_in=100 --dim_out=100 --ood=3.0 --ood_test=-3.0 --data_version=3 &
# python synCBIV_OOD.py --iter=8000 --lrate=0.002 --num_reps=10 --des_str='/0114_lr=0.002_iter=8000_dim=128/' --dim_in=128 --dim_out=128 --ood=3.0 --ood_test=-3.0 --data_version=3 &
# python synCBIV_OOD.py --iter=8000 --lrate=0.002 --num_reps=10 --des_str='/0114_lr=0.002_iter=8000_dim=256/' --dim_in=256 --dim_out=256 --ood=3.0 --ood_test=-3.0 --data_version=3 &
# python synCBIV_OOD.py --iter=8000 --lrate=0.002 --num_reps=10 --des_str='/0114_lr=0.002_iter=8000_dim=384/' --dim_in=384 --dim_out=384 --ood=3.0 --ood_test=-3.0 --data_version=3 &
# python synCBIV_OOD.py --iter=8000 --lrate=0.002 --num_reps=10 --des_str='/0114_lr=0.002_iter=8000_dim=512/' --dim_in=512 --dim_out=512 --ood=3.0 --ood_test=-3.0 --data_version=3 &

# python synCBIV_OOD.py --iter=20000 --lrate=0.002 --num_reps=10 --des_str='/0114_lr=0.002_iter=20000_dim=100/' --dim_in=100 --dim_out=100 --ood=3.0 --ood_test=-3.0 --data_version=2 &
# python synCBIV_OOD.py --iter=20000 --lrate=0.002 --num_reps=10 --des_str='/0114_lr=0.002_iter=20000_dim=128/' --dim_in=128 --dim_out=128 --ood=3.0 --ood_test=-3.0 --data_version=2 &
# python synCBIV_OOD.py --iter=20000 --lrate=0.002 --num_reps=10 --des_str='/0114_lr=0.002_iter=20000_dim=256/' --dim_in=256 --dim_out=256 --ood=3.0 --ood_test=-3.0 --data_version=2 &
# python synCBIV_OOD.py --iter=20000 --lrate=0.002 --num_reps=10 --des_str='/0114_lr=0.002_iter=20000_dim=384/' --dim_in=384 --dim_out=384 --ood=3.0 --ood_test=-3.0 --data_version=2 &
# python synCBIV_OOD.py --iter=20000 --lrate=0.002 --num_reps=10 --des_str='/0114_lr=0.002_iter=20000_dim=512/' --dim_in=512 --dim_out=512 --ood=3.0 --ood_test=-3.0 --data_version=2 &

# 0115
# python synCBIV_OOD.py --iter=8000 --lrate=0.002 --num_reps=12 --des_str='/0115_lr=0.002_iter=8000_dim=256/' --dim_in=256 --dim_out=256 --ood=3.0 --ood_test=-3.0 --data_version=4 &
# python synCBIV_OOD.py --iter=20000 --lrate=0.002 --num_reps=12 --des_str='/0115_lr=0.002_iter=20000_dim=256/' --dim_in=256 --dim_out=256 --ood=3.0 --ood_test=-3.0 --data_version=4 &
# python synCBIV_OOD.py --iter=20000 --lrate=0.001 --num_reps=12 --des_str='/0115_lr=0.001_iter=20000_dim=256/' --ood=3.0 --ood_test=-3.0 --data_version=4 &

# python synCBIV_OOD.py --iter=8000 --lrate=0.002 --num_reps=20 --des_str='/0115_lr=0.002_iter=8000_elu/' --nonlin='elu' --ood=3.0 --ood_test=-3.0 --data_version=2 --use_gpu=0 &
# python synCBIV_OOD.py --iter=8000 --lrate=0.002 --num_reps=20 --des_str='/0115_lr=0.002_iter=8000_relu/'  --nonlin='relu' --ood=3.0 --ood_test=-3.0 --data_version=2 --use_gpu=0 &