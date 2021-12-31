# Adaptive Modeling Against Adversarial Attacks

This is the official code release for the [paper](https://arxiv.org/abs/2112.12431) "Adaptive Modeling Against Adversarial Attacks".

* Please note that the algorithm might be referred as **post training** for easy reference.

## Envrionment Setups
We recommend using Anaconda/Miniconda to setup the envrionment, with the following command:
```bash
conda env create -f pt_env.yml
conda activate post_train
```

## Experiments
We have mainly conducted our experiment on two base model structures: [Fast FGSM](https://arxiv.org/abs/2001.03994) and [Madry Model](https://arxiv.org/abs/1706.06083). 
Experiments are based on **CIFAR-10** and **MNIST** datasets. 

To reproduce the experiment results on these two models, you can refer to the following repositories for more details:

Fast FGSM: https://github.com/JokerYan/fast_adversarial.git

Madry Model: https://github.com/JokerYan/pytorch-adversarial-training.git 
