#! /bin/sh

#PBS -q volta_gpu
#PBS -j oe
#PBS -N pytorch
#PBS -l select=1:ncpus=10:mem=80gb:ngpus=1
#PBS -l walltime=02:00:00
#PBS -P 11002407

cd $PBS_O_WORKDIR;
np=$(cat ${PBS_NODEFILE} | wc -l);
#image="/app1/common/singularity-img/3.0.0/pytorch_1.9_cuda_11.3.0-ubuntu20.04-py38-ngc_21.04.simg"
image="/app1/common/singularity-img/3.0.0/pytorch_1.11_cuda_11.3_cudnn8-py38.sif"


singularity exec $image bash << EOF > stdout.$PBS_JOBID 2> stderr.$PBS_JOBID
export PYTHONPATH=$PYTHONPATH:/home/svu/e0200920/volta_pypkg/lib/python3.8/site-packages

python train.py --pretrained_model --num_epochs 50 --method=$method --dataset=$dataset --spurious_strength=$spurious_strength --val_target_size=$val_target_size --weight_decay=$weight_decay --batch_size=$batch_size --init_lr=$init_lr --seed=$seed --data_dir=/hpctmp/e0200920/mcdominoes_data --output_dir=/hpctmp/e0200920/$method-$dataset-$spurious_strength-$val_target_size-$weight_decay-$batch_size-$init_lr-$seed

EOF
