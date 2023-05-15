#! /bin/sh

method=2
dataset="mcdominoes"
spurious_strength=1
val_target_size=2000
regularize_mode=0

weight_decay_array=(1e-1 1e-2 1e-3 1e-4)
batch_size_array=(8 16 32 64)
lr_array=(1e-2 1e-3 1e-4)
group_size_array=(1 4 16 64)


for i in {1..30}
do
	echo $i
	weight_decay=${weight_decay_array[$(( RANDOM % 4 ))]}
	batch_size=${batch_size_array[$(( RANDOM % 4 ))]}
	init_lr=${lr_array[$(( RANDOM % 3 ))]}
	group_size=${group_size_array[$(( RANDOM % 4 ))]}
	exp_name=$method-$regularize_mode-$dataset-$spurious_strength-$val_target_size-$weight_decay-$batch_size-$group_size-$init_lr-$seed
	echo $exp_name
	for seed in 0 1 2
	do
		qsub -v method=$method,regularize_mode=$regularize_mode,dataset=$dataset,spurious_strength=$spurious_strength,val_target_size=$val_target_size,weight_decay=$weight_decay,batch_size=$batch_size,init_lr=$init_lr,group_size=$group_size,seed=$seed submit_2.sh
	done
done
