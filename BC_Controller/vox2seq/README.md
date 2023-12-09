# Voxel2Sequence

This module is to get the order about placing blocks.

### 1.Install independencies

```bash
pip install -r requirements
```

### 2.Preprocess dataset

You need to process dataset to get voxel-action pairs that we can use to train.

```bash
python preprocess.py
```


### 3.Train

```bash
# you can modify some hyper-parameters
python train.py
```

### 4.Export model

```bash
# export .ckpt file to .pt file about model weights
python save.py
```
