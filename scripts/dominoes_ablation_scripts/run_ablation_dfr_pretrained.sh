#! /bin/sh

dataset="mcdominoes"

for spurious_strength in 0.95 0.96 0.97 0.98 0.99 1
do
	for val_target_size in 1000 2000 3000 4000 5000 6000
	do
		for seed in 0 1 2
		do
			exp_name=$dataset-$spurious_strength-$val_target_size-$seed
        		echo $exp_name
                        filename="$i"".venus01"
			qsub -v dataset=$dataset,spurious_strength=$spurious_strength,val_target_size=$val_target_size,seed=$seed submit_dfr_pretrained.sh
		done
	done
done
