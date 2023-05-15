#! /bin/sh

dataset="celeba"
val_target_size=2000
spurious_strength=1

for weight_decay in 1e-1 1e-2 1e-3 1e-4
do
	for batch_size in 8 16 32 64
	do
		for init_lr in 1e-2 1e-3 1e-4
		do
			for seed in 0 1 2
			do
				exp_name=$method-$dataset-$spurious_strength-$val_target_size-$weight_decay-$batch_size-$init_lr-$seed
        			echo $exp_name
                        	filename="$i"".venus01"
				qsub -v dataset=$dataset,spurious_strength=$spurious_strength,val_target_size=$val_target_size,weight_decay=$weight_decay,batch_size=$batch_size,init_lr=$init_lr,seed=$seed submit_dfr.sh
			done
		done
	done
done
