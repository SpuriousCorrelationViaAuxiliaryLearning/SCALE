#! /bin/sh

method=0
dataset="mcdominoes"

weight_decay=1e-3
batch_size=16
init_lr=1e-3

for spurious_strength in 0.95 0.96 0.97 0.98 0.99 1
do
	for val_target_size in 1000 2000 3000 4000 5000 6000
	do
		for seed in 0 1 2
		do
			exp_name=$method-$dataset-$spurious_strength-$val_target_size-$weight_decay-$batch_size-$init_lr-$seed
        		echo $exp_name
			qsub -v method=$method,dataset=$dataset,spurious_strength=$spurious_strength,val_target_size=$val_target_size,weight_decay=$weight_decay,batch_size=$batch_size,init_lr=$init_lr,seed=$seed submit_0.sh
		done
	done
done
