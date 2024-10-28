# Data-free knowledge distillation via generator-free data generation for Non-IID federated learning (FedF2DG)

This is the official pytorch implementation of [Data-free knowledge distillation via generator-free data generation for Non-IID federated learning]. The link to the paper is (https://www.sciencedirect.com/science/article/pii/S0893608024005513?fr=RR-2&ref=pdf_download&rr=8d9979bd0aa1ddc4).


If you find our work is useful for your work, please kindly cite our paper.

```
@article{zhao2024data,
  title={Data-free knowledge distillation via generator-free data generation for Non-IID federated learning},
  author={Zhao, Siran and Liao, Tianchi and Fu, Lele and Chen, Chuan and Bian, Jing and Zheng, Zibin},
  journal={Neural Networks},
  volume={179},
  pages={106627},
  year={2024},
  publisher={Elsevier}
}
```

## Framework


![overview](./overview.png)



## Run the script

### 1. run federated_learning_stage.sh
Pre-train model parameters
```
sh federated_learning_stage.sh
```
```bash
nohup python -u experiments.py --model=resnet \
    --dataset=cifar10 \
    --alg=scaffold \
    --lr=0.01 \
    --batch-size=64 \
    --epochs=5 \
    --n_parties=10 \
    --mu=0.01 \
    --rho=0.9 \
    --comm_round=20 \
    --reg=1e-5 \
    --partition=noniid-labeldir \
    --beta=0.05 \
    --device='cuda:0' \
    --datadir='/home/zsr/data/' \
    --logdir='./logs/test/' \
    --noise=0 \
    --sample=1 \
    --label_noise_type='None' \
    --label_noise_rate=0.0 \
    --init_seed=0 >nohup.out&
```
### 2. run adaptive_data_generation_stage.sh
```
sh adaptive_data_generation_stage.sh
```
```bash
nohup python -u experiments.py --model=resnet \
    --dataset=cifar10 \
    --alg=adaptive_data_generation \
    --lr=0.01 \
    --batch-size=64 \
    --n_parties=10 \
    --mu=0.01 \
    --rho=0.9 \
    --reg=1e-5 \
    --partition=noniid-labeldir \
    --beta=0.05 \
    --device='cuda:0' \
    --datadir='/home/zsr/data/' \
    --logdir='./logs/test/' \
    --noise=0 \
    --sample=1 \
    --label_noise_type='None' \
    --label_noise_rate=0.0 \
    --bs=256 \
    --num_batch=10 \
    --cig_scale=0.0 \
    --init_seed=0 >nohup.out&
```
### 3. run knowledge_distillation_stage.sh
```
sh knowledge_distillation_stage.sh
```
```bash
nohup python -u experiments.py --model=resnet \
    --dataset=cifar10 \
    --alg=knowledge_distillation \
    --batch-size=64 \
    --n_parties=10 \
    --mu=0.01 \
    --rho=0.9 \
    --reg=1e-3 \
    --partition=noniid-labeldir \
    --beta=0.05\
    --device='cuda:0' \
    --logdir='./logs/test/' \
    --noise=0 \
    --sample=1 \
    --label_noise_type='None' \
    --label_noise_rate=0.0 \
    --kd_epochs = 30\
    --kd_lr = 0.01\
    --weight_decay = 1e-4\
    --init_seed=0 >nohup.out&
```
### 4. run Rcompete.sh
Generate diverse samples and perform knowledge distillation.
```
sh Rcompete.sh
```
```bash
nohup python -u experiments.py --model=resnet \
    --dataset=cifar10 \
    --alg=knowledge_distillation_Rcompete \
    --lr=0.01 \
    --batch-size=64 \
    --epochs=5 \
    --n_parties=10 \
    --mu=0.01 \
    --rho=0.9 \
    --comm_round=10 \
    --reg=1e-5 \
    --partition=noniid-labeldir \
    --beta=0.05 \
    --device='cuda:0' \
    --datadir='/home/zsr/data/' \
    --logdir='./logs/test/' \
    --noise=0 \
    --sample=1 \
    --label_noise_type='None' \
    --label_noise_rate=0.0 \
    --bs=256 \
    --num_batch=1 \
    --cig_scale=10.0 \
    --kd_epochs = 30\
    --kd_lr = 0.01\
    --weight_decay = 1e-4\
    --init_seed=0 >nohup.out&
```
## Parameters Description

#### 1. Basic Parameters

>+ **init_seed**: seed for reproducibility, default is 0
>+ **alg**: traning method, choices in {'fedavg', 'fedprox', 'scaffold', 'fednova', 'moon'}, default is 'fedavg'
>+ **dataset**: training dataset, choices in {'CIFAR10', 'CIFAR100','SVHN'}, default is 'CIFAR10'
>+ **lr**: client learning rate, default is 0.01
>+ **epochs**: number of local epochs, default is 5
>+ **n_parties**: the number of the clients, default is 100
>+ **sample**: sample ratio for each communication round
>+ **comm_round**: number of communication rounds, default is 100
>+ **datadir**: data directory path
>+ **logdir**: log directory path
>+ **beta**: the parameter for the dirichlet distribution for data partitioning
>+ **mu**: mu parameter in FedProx and MOON, default is 1e-3

#### 2. Data Generation Parameters
>+ **bs**: batch size of generated data
>+ **num_batch**: number of batch
>+ **iters_mi**: number of iterations for model inversion
>+ **cig_scale**: competition score
>+ **competition score**: lr for deep inversion
>+ **di_var_scale**: TV L2 regularization coefficient
>+ **di_l2_scale**: weight for BN regularization statistic

#### 3. Knowledge Distillation Parameters
>+ **kd_epochs**: number of total epochs to run
>+ **kd_lr**: kd learning rate
>+ **weight_decay**: weight decay

