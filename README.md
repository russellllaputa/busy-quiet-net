# Busy-Quiet Video Disentangling for Video Classification [[arXiv]](https://arxiv.org/abs/2103.15584)


## Prerequisites
- [PyTorch](https://pytorch.org/) 1.3 or higher

## Data Preparation

Please refer to [TSM](https://github.com/mit-han-lab/temporal-shift-module) repo for the details of data preparation.


## Pretrained Models

Coming soon

Because of some modifications of the filenames in our project, many trained models cannot be resumed by the current code. 
We are re-training the models.
By running the training scripts we provided bellow, you should get similar results as stated in the paper.

## Training 

We provided several examples to train BQN with this repo:

- To train on Kinetics, you can run the script bellow:
 ```bash
 
 python main_apex.py --modality HP --dataset kinetics --dense_sample --root-model kinetics --arch resnet50 --dist-url "tcp://ip_address:6065"   --dist-backend 'nccl' \
 --multiprocessing-distributed  --available_gpus 0,1,2,3 --npb --lr 0.08 --wd 2e-4 --dropout 0.5 --num_segments 16  --batch_size 32 --batch_multiplier 1 \
 --lr_type cos --epochs 100 --warmup_epochs 7 -j 28 --gd 20 --ls --suffix 1  --prefix BQN --world-size 16 --rank  0

 python main_apex.py --modality HP --dataset kinetics --dense_sample --root-model kinetics --arch x3dm --dist-url "tcp://ip_address:6065"   --dist-backend 'nccl' \
 --multiprocessing-distributed  --available_gpus 0,1,2,3 --npb --lr 0.4 --wd 5e-5 --dropout 0.5 --num_segments 16  --batch_size 64 --batch_multiplier 1 \
 --lr_type cos --epochs 256 --warmup_epochs 7 -j 28 --gd 20 --ls --suffix 1  --prefix BQN --world-size 4 --rank  0
 
```

- To train on Something-Something V1, you can run the script bellow:
 ```bash
 python main_apex.py --modality HP --dataset something --root-model something --arch x3dm --dist-url "tcp://ip_address:6065"   --dist-backend 'nccl' \
 --multiprocessing-distributed  --available_gpus 0,1,2,3 --npb --lr 0.2   --wd 5e-5 --dropout 0.5 --num_segments 16  --batch_size 64 --batch_multiplier 1 \
 --lr_type cos --epochs 100 --warmup_epochs 7 -j 28 --gd 20 --ls --suffix 1  --prefix BQN --world-size 4 --rank  0

 python main_apex.py --modality HP --dataset something --root-model something --arch resnet50 --dist-url "tcp://ip_address:6065"   --dist-backend 'nccl' \
 --multiprocessing-distributed  --available_gpus 0,1,2,3 --npb --lr 0.12   --wd 8e-4 --dropout 0.8 --num_segments 16 --batch_size 64 --batch_multiplier 1 \
 --lr_type cos --epochs 50 --warmup_epochs 7 -j 28 --gd 20 --ls --suffix 1  --prefix BQN --world-size 4 --rank  0
 
# Notice that the total batch size is equal to batch_size x batch_multiplier x world_size, and 
# you should scale up the learning rate with batch size. For example, if you use 
# a batch size of 128 you should set learning rate to 0.4.
  ```
  


## Test 

For example, to test the models, you can run the scripts below. The scripts test on 16-frame setting by running:

```bash
# test on kinetics
python test_f.py something --weights=ckpt.pt --test_segments=16 --batch_size=4  -j 10 --test_crops=3  --full_res --dense_sample
# test on Something
python test_f.py something --weights=ckpt.pt --test_segments=16 --batch_size=4  -j 10 --test_crops=3  --full_res --twice_sample

```

## Other Info

### References

This repository is built upon the following baseline implementations.

- [TSM](https://github.com/mit-han-lab/temporal-shift-module)
- [PyTorchVideo](https://github.com/facebookresearch/pytorchvideo)


### Contact

For any questions, please feel free to open an issue or contact:

```
Guoxi Huang: gh825@york.ac.uk
```
