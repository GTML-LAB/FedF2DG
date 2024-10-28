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

####1. run federated_learning_stage.sh
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
####2. run adaptive_data_generation_stage.sh
```
sh adaptive_data_generation_stage.sh
```
```bash
nohup python -u experiments.py --model=resnet \
    --dataset=cifar10 \
    --alg=Adaptive Data Generation \
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
## Parameters Description

#### 1. Basic Parameters


>+ **seed**: seed for reproducibility, default is 1024
>+ **method**: traning method, choices in {'FedAvg', 'FedProx', 'FedDyn', 'SCAFFOLD', 'MOON', 'FedFTG', 'FedProxGAN', 'FedDynGAN', 'SCAFFOLDGAN', 'MOONGAN'}, default is 'FedDyn'
>+ **dataset**: training dataset, choices in {'CIFAR10', 'CIFAR100'}, default is 'CIFAR10'
>+ **exp_name**: experiment name, input whatever you want, default is 'Federated'
>+ **save**: bool value for saving the training results or not, default is False
>+ **savepath**: directory to save exp results, default is 'result/'
>+ **print_freq**: print info frequency(ACC) on each client locally, default is 2
>+ **save_period**: the frequency of saving the checkpoint, default is 200

#### 2. Data Segmentation Parameters

>+ **n_client**: the number of the clients, default is 100
>+ **rule**: split rule of dataset, choices in {iid, Dirichlet}
>+ **alpha**: control the non-iidness of dataset, the parameter of Dirichlet, default is 0.6. Please ignore this parameter if rule is 'iid'
>+ **sgm**: the unbalanced parameter by using lognorm distribution, sgm=0 indicates balanced

#### 3. Training Parameters

>+ **localE**: number of local epochs, default is 5
>+ **comm_amount**: number of communication rounds, default is 1000
>+ **active_frac**: the fraction of active clients per communication round, default is 1.0, indicating all the clients participating in the communications
>+ **bs**: batch size on each client, default is 50
>+ **n_minibatch**: the number of minibatch size in SCAFFOLD, default is 50
>+ **lr**: client learning rate, default is 0.1
>+ **momentum**: local (client) momentum factor, default is 0.0
>+ **weight_decay**: local (client) weight decay factor, default is 1e-3
>+ **lr_decay**: local (client) learning rate decay factor, default is 0.998
>+ **coef_alpha**:alpha coefficient in FedDyn, default is 1e-2
>+ **mu**: mu parameter in FedProx and MOON, default is 1e-4
>+ **tau**: mu parameter in MOON, default is 1
>+ **sch_step**: the learning rate scheduler step, default is 1
>+ **sch_gamma**: the learning rate scheduler gamma, default is 1.0

