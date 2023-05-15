# Spurious Correlation Via Auxiliary LEarning (SCALE)


## Data access
The CelebA dataset has to be manually downloaded from  [here](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) and moved into the `data/celeba` directory.

The MNIST-CIFAR Dominoes dataset will be automatically downloaded and generated.


## Example commands: 

### Base models

To train base ERM models on the Dominoes dataset, use the command:
```bash
python train.py --pretrained_model --num_epochs 50 --method=0 --dataset=mcdominoes \
  --spurious_strength=1 --val_target_size=2000 --weight_decay=1e-3 --batch_size=16 \
   --init_lr=1e-3 --seed=0 --data_dir=<DOMINOES_DIR> --output_dir=<OUTPUT_DIR>
```

The `OUTPUT_DIR` is a path to the folder where the logs will be stored.
The `DOMINOES_DIR` is the directory to download and store the dominoes dataset.

The `--pretrained_model` flag initializes the model with the 
ImageNet-pretrained weights.
The number of epochs, weight decay, learning rate and batch size is set via 
the `--num_epochs`, `--weight_decay`, `--init_lr`, and `--batch_size` flags
respectively.

The `--spurious_strength` flag controls the strength of the spurious correlation in 
the train dataset, with 1 corresponding to 100% correlation. The `--val_target_size` flag controls the size of the validation + target size, with the
target size = val_target_size / 2. The `--dataset` flag determines which dataset to use, which can be either mcdominoes or celeba

The `--method` flag determines which method to use to train the model, which can take values of
0, 1, 2, which corresponds to ERM on train, ERM on target, 
and our proposed SCALE method respectively.

### DFR

The DFR technique retrains the last-layer of the model on the target dataset, 
which is used on the model learnt from the ERM on train method (Method 0).
```bash
python retrain.py --dataset=mcdominoes --val_target_size=2000 
--spurious_strength=1 --ckpt_path=<CKPT_PATH> --seed=0 
--data_dir=<DOMINOES_DIR> --output_dir=<OUTPUT_DIR>
```
The `CKPT_PATH` is the directory of the model to be retrained, and the `OUTPUT_DIR` is the 
directory where the retrained model is saved.

The `--seed`, `--spurious_strength`, and `--val_target_size` flags should be the same as
the flags that the model in the `CKPT_PATH` was trained on. This is to ensure that the train,
target, validation, and test datasets do not change, which would otherwise result in data leakage.

### SCALE
The SCALE method uses both the train and target gradients for the model updates, and 
additionally regularizes the train gradients based on the gradient alignment between the
train and the target gradients, measured by the cosine similarity.
```bash
\python train.py --pretrained_model --num_epochs 50 --method=2 --dataset=mcdominoes \
  --spurious_strength=1 --val_target_size=2000 --weight_decay=1e-3 --batch_size=16 \
   --init_lr=1e-3 --group_size=64 --regularize_mode=0 --seed=0 
   --data_dir=<DOMINOES_DIR> --output_dir=<OUTPUT_DIR>
```
Our proposed method has two extra hyperparameters. The  `--group_size` flag sets the size of
each parameter group, since the cosine similarity and task weight parameter is 
computed and applied in a group-wise manner.

The `--regularize_mode` determines the function that maps the cosine similarity to the 
task weight parameter, with `--regularize_mode=0` and `--regularize_mode=3` corresponding to
Linear and Step functions respectively.

## Additional Scripts
The scripts used to run the tuning and ablation experiments can be found in the
`scripts` folder.

## References
This code is modified from the original [_Deep Feature Reweighting_](https://github.com/PolinaKirichenko/deep_feature_reweighting) codebase.

The code for the generation of the Dominoes dataset is modified from the original [_Simplicity Bias_](https://github.com/harshays/simplicitybiaspitfalls) codebase.
