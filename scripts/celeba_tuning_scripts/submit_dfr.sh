#! /bin/sh

#PBS -q volta_gpu
#PBS -j oe
#PBS -N pytorch
#PBS -l select=1:ncpus=10:mem=80gb:ngpus=1
#PBS -l walltime=1:30:00
#PBS -P 11002407

cd $PBS_O_WORKDIR;
np=$(cat ${PBS_NODEFILE} | wc -l);
#image="/app1/common/singularity-img/3.0.0/pytorch_1.9_cuda_11.3.0-ubuntu20.04-py38-ngc_21.04.simg"
image="/app1/common/singularity-img/3.0.0/pytorch_1.11_cuda_11.3_cudnn8-py38.sif"


singularity exec $image bash << EOF > stdout.$PBS_JOBID 2> stderr.$PBS_JOBID
export PYTHONPATH=$PYTHONPATH:/home/svu/e0200920/volta_pypkg/lib/python3.8/site-packages

python retrain.py --dataset=$dataset --val_target_size=$val_target_size --spurious_strength=$spurious_strength --ckpt_path=/hpctmp/e0200920/method_0_tuning/0-$dataset-$spurious_strength-$val_target_size-$weight_decay-$batch_size-$init_lr-$seed/best_checkpoint.pt --seed=$seed --data_dir=/hpctmp/e0200920/celeba --output_dir=/hpctmp/e0200920/DFR-$dataset-$spurious_strength-$val_target_size-$weight_decay-$batch_size-$init_lr-$seed


EOF
