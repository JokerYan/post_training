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


## How to Use
The post training algorithm can be applied to adversarially trained models at inference stage.
Before doing inference of any input, post train the model first to adapt to the input. 
Then use the post trained model to infer the input.

### Setups
You will need to prepare a few items to run the post training algorithm:
* Model: post trained model with parameter loaded
* Train_Loader: dataloader that loads the training dataset
* Train_Loaders_By_Class: list of dataloaders, where the i_th dataloader contains only the data from the i_th class. 
  You can prepare this list by calling `post_training.get_train_loaders_by_class()`
* Test_Loader: dataloader that loads the test dataset. **Important**: post training only accepts test input of batch size 1.
* Args: arguments used to configure the post training algorithm, which can be prepared by calling `arguments.get_args()`. 
  Detailed explanation of each argument is in the later section.

You can refer to the following code snippet as an example:
```python
import torch
from arguments import get_args
from post_training import get_train_loaders_by_class
from utils import get_datasets, get_loaders
from models import PreActResNet18

args = get_args()
train_dataset, test_dataset = get_datasets()
train_loader, test_loader = get_loaders(train_batch_size=128, test_batch_size=1)
train_loaders_by_class = get_train_loaders_by_class(train_dataset, batch_size=128)

model = PreActResNet18().cuda()
state_dict = torch.load(pretrained_model_path)
model.load_state_dict(state_dict)
```

### Inference
During the inference stage, base model is post trained to adapt to the adversarial/natural input.
The resultant model is the used to infer the input and return the prediction.

You can refer to the following code snippet as an example:
```python
import torch
from post_training import post_train

for i, (data, label) in enumerate(test_loader):
    with torch.no_grad():
        post_model = post_train(model, data, train_loader, train_loaders_by_class, args)
        output = post_model(data)
        pred = torch.argmax(output)
```

### Arguments
The arguement description and accepted values are listed here:
* pt-data: post training data composition
  - ori_rand: 50% original class + 50% random class
  - ori_neigh: 50% original calss + 50% neighbour class
  - train: random training data
* pt-method: post training method
  - adv: fast adversarial training used in Fast FGSM
  - dir_adv: fixed adversarial training proposed in paper
  - normal: normal training instead of adversarial training
* adv-dir: direction of fixed adversarial training
  - na: not applicable, used for adv and normal pt-method
  - pos: positive direction, data + fix perturbation
  - neg: negative direction, data - fix perturbation
  - both: default for dir_adv, random mixture of positive and negative direction
* neigh-method: attack method to find the neighbour
  - untargeted: use untargeted attack
  - targeted: use targeted attack and choose the highest confidence class
* pt-iter: post training iteration
* pt-lr: post training learning rate
* att-iter: attack iteration used for attack and post adversarial training
* att-restart: attack restart used for attack and post adversarial training
* log-file: log file stored path
