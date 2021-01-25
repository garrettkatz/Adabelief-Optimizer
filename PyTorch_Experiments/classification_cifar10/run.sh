# # orig:
# CUDA_VISIBLE_DEVICES=0 python3 main.py --optim adabelief --lr 1e-3 --eps 1e-8 --beta1 0.9 --beta2 0.999 --momentum 0.9

# GK (reruns from curve names):
# # resnet-sgd-lr0.1-momentum0.9-wdecay0.0005-run0-resetFalse
# CUDA_VISIBLE_DEVICES=0 python3 main.py --model resnet --optim sgd --lr 1e-1 --momentum 0.9 --weight_decay 0.0005

# # resnet-capb-lr0.1-momentum0.9-wdecay0.0005-run0-resetFalse
# CUDA_VISIBLE_DEVICES=0 python3 main.py --model resnet --optim capb --lr 1e-1 --momentum 0.9 --weight_decay 0.0005

# resnet-adabelief-lr0.001-betas0.9-0.999-eps1e-08-wdecay0.0005-run0-resetFalse
# CUDA_VISIBLE_DEVICES=0 python3 main.py --model resnet --optim adabelief --lr 1e-3 --eps 1e-8 --beta1 0.9 --beta2 0.999 --momentum 0.9 --weight_decay 0.0005

# # resnet-abcapb-lr0.001-betas0.9-0.999-eps1e-08-wdecay0.0005-run0-resetFalse
# CUDA_VISIBLE_DEVICES=0 python3 main.py --model resnet --optim abcapi --lr 1e-3 --eps 1e-8 --beta1 0.9 --beta2 0.999 --momentum 0.9 --weight_decay 0.0005

# # resnet-abcapi-cl100-lr0.001-betas0.9-0.999-eps1e-08-wdecay0.0005-run0-resetFalse
# CUDA_VISIBLE_DEVICES=0 python3 main.py --classes 100 --model resnet --optim abcapi --lr 1e-3 --eps 1e-8 --beta1 0.9 --beta2 0.999 --momentum 0.9 --weight_decay 0.0005

# trying more aggressive decay_epoch 60, total_epoch 110
# resnet-abcapi-cl100-lr0.001-betas0.9-0.999-eps1e-08-wdecay0.0005-run0-resetFalse
CUDA_VISIBLE_DEVICES=0 python3 main.py --classes 100 --model resnet --optim abcapi --lr 1e-3 --eps 1e-8 --beta1 0.9 --beta2 0.999 --momentum 0.9 --weight_decay 0.0005 --decay_epoch 60 --total_epoch 200
