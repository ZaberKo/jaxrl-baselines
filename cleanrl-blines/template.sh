#!/bin/bash
#SBATCH -o job.%j.out          # 脚本执行的输出将被保存在当job.%j.out文件下，%j表示作业号;
#SBATCH --partition=gpulab01      # 作业提交的指定分区队列为titan
#SBATCH --qos=gpulab01            # 指定作业的QOS
#SBATCH -J lh-jaxrl-baselines-job       # 作业在调度系统中的作业名为myFirstJob;
#SBATCH --nodes=1              # 申请节点数为1,如果作业不能跨节点(MPI)运行, 申请的节点数应不超过1
#SBATCH --ntasks-per-node=6    # 每个节点上运行一个任务，默认一情况下也可理解为每个节点使用一个核心；
#SBATCH --gres=gpu:1           # 指定作业的需要的GPU卡数量，集群不一样，注意最大限制; 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lh_cse@163.com

# 设置 WandB API key
# export WANDB_API_KEY="5341b8b0aa7a3635f2ea7c71e238ce16b52dfd9b" 

python train.py --config-name dqn

python train.py --config-name ddpg

python train.py --config-name td3

python train.py --config-name c51