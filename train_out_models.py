from torchvision.datasets import CIFAR10, CIFAR100, GTSRB
import torch.nn as nn
import torch
import random
import torch.nn.functional as F
from random import sample
import torchvision
import torch.optim as optim
import numpy as np
from utils.TorchMLUtils import TorchMLUtils as tml
from torchvision import transforms as T
from utils.adaptivepoison import AdaptivePoison as AP
import argparse

def train_out_models(data_str, n_sms, psn_thresh, max_psn_iters, cuda_device):
    
    if((data_str == 'cifar10') or (data_str == 'cifar100')):
        train_transform = T.Compose([T.ToTensor(), 
                                     T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                     T.RandomHorizontalFlip(),
                                     T.RandomCrop(32, padding=4, padding_mode="reflect"),
                                    ]) 
        test_transform = T.Compose([T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        
    elif(data_str == 'gtsrb'):
        train_transform = T.Compose([
                                T.Resize((32, 32)),
                                T.ToTensor(),
                                T.RandomHorizontalFlip(),
                                T.RandomCrop(32, padding=4, padding_mode="reflect"),
                                T.Normalize((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669))
                                ])
        
        test_transform = T.Compose([
                            T.Resize((32, 32)),
                            T.ToTensor(),
                            T.Normalize((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669))
                            ])
    
    if(data_str == 'cifar10'):
        original_dataset = CIFAR10(root="CIFAR10", train=True, transform=train_transform, download=True)
        num_labels = 10
        # save_dir = "CIFAR10-"+str(n_poisons)+"PModels"
        num_samples = 25000
    
    elif(data_str == 'cifar100'):
        original_dataset = CIFAR100(train=True, download=True, root="CIFAR100", transform=train_transform)
        num_labels = 100
        # save_dir = "CIFAR100-"+str(n_poisons)+"PModels"
        num_samples = 25000
    
    elif(data_str == 'gtsrb'):
        original_dataset = GTSRB(root="GTSRB", split='train', transform=train_transform, download=True)
        num_labels = 43
        # save_dir = "GTSRB-"+str(n_poisons)+"PModels"
        num_samples = 13320
    

    model = torchvision.models.resnet18(pretrained = False, num_classes = num_labels)
  
    ap_helper = AP(experiment_name="LabelOnly", 
                 model=model, 
                 output_dim=num_labels, 
                 dataset=original_dataset, 
                 criterion=nn.CrossEntropyLoss(),
                 optimizer = torch.optim.SGD,
                 train_size=num_samples,
                 holdout_size=0, 
                 num_out_models = n_sms,
                 poison_thresh = psn_thresh,
                 max_poison_iters = max_psn_iters,
                 num_target_points = 500,
                 train_transform = train_transform,
                 test_transform = test_transform,
                 epochs=100, 
                 lr=0.1, 
                 batch_size=256,
                 dataset_name = data_str.upper(),
                 load_saved=False,
                 seed = 42,
                 device=cuda_device)
    
    ap_helper.run_poison_phase()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '-d',
        '--data',
        help='dataset name',
        type=str,
        default='cifar10'
    )
    
    parser.add_argument(
        '-out',
        '--outmodels',
        help='num of OUT models',
        type=int,
        default= 8
    )
    
    parser.add_argument(
        '-tp',
        '--psnthresh',
        help='poison threshold',
        type=float,
        default= 0.13
    )
    
    parser.add_argument(
        '-kmax',
        '--maxiters',
        help='maximum poison iterations',
        type=int,
        default= 6
    )
    
    
    parser.add_argument(
        '-c',
        '--cuda',
        help='gpu device number',
        type=int,
        default=0
    )
    
    arguments = vars(parser.parse_args())
    data_str = arguments['data'].lower()
    n_sms = 2*arguments['outmodels']
    psn_thresh = arguments['psnthresh']
    max_psn_iters = arguments['maxiters']
    cuda_device = 'cuda:' + str(arguments['cuda'])
    
    train_out_models(data_str,n_sms, psn_thresh, max_psn_iters, cuda_device)
        
   
