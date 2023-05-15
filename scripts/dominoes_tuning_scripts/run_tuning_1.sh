#! /bin/sh

method=1
dataset="mcdominoes"
spurious_strength=1
val_target_size=2000

weight_decay_array=(1e-1 1e-2 1e-3 1e-4)
batch_size_array=(8 16 32 64)
lr_array=(1e-2 1e-3 1e-4)


for i in {1..30}
do
	echo $i
	weight_decay=${weight_decay_array[$(( RANDOM % 4 ))]}
	batch_size=${batch_size_array[$(( RANDOM % 4 ))]}
	init_lr=${lr_array[$(( RANDOM % 3 ))]}
	exp_name=$method-$dataset-$spurious_strength-$val_target_size-$weight_decay-$batch_size-$init_lr-$seed
	echo $exp_name
	for seed in 0 1 2
	do
		qsub -v method=$method,dataset=$dataset,spurious_strength=$spurious_strength,val_target_size=$val_target_size,weight_decay=$weight_decay,batch_size=$batch_size,init_lr=$init_lr,seed=$seed submit_1.sh
	done
done
