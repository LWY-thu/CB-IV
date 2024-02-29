# 0225
# python synCBIV_OOD.py --iter=20000 --lrate=0.002 --num_reps=20 --des_str='/0225_lr=0.002_iter=20000/' --ood=3.0 --ood_test=-3.0 --data_version=5 &

# 0226 测试10000:1000数据实验
# python synCBIV_OOD_Aggr_v1.py --iter=20000 --lrate=0.002 --num_reps=20 --des_str='/0226_e2_ratio=0.1/' --e_second_ratio=0.1 --data_version=5 &
# python synCBIV_OOD_Aggr_v1.py --iter=20000 --lrate=0.002 --num_reps=20 --des_str='/0226_e2_ratio=0.05/' --e_second_ratio=0.05 --data_version=5 &

python synCBIV_OOD_Aggr_v1.py --iter=20000 --lrate=0.002 --num_reps=20 --des_str='/0226_e2_ratio=0.1/' --e_second_ratio=0.1 --data_version=2 &
python synCBIV_OOD_Aggr_v1.py --iter=20000 --lrate=0.002 --num_reps=20 --des_str='/0226_e2_ratio=0.05/' --e_second_ratio=0.05 --data_version=2 &

python synCBIV_OOD_Aggr_v1.py --iter=20000 --lrate=0.002 --num_reps=20 --des_str='/0226_e2_ratio=0.1/' --e_second_ratio=0.1 --data_version=3 &
python synCBIV_OOD_Aggr_v1.py --iter=20000 --lrate=0.002 --num_reps=20 --des_str='/0226_e2_ratio=0.05/' --e_second_ratio=0.05 --data_version=3 &