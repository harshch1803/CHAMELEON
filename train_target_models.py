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
from utils.labelonlymi import LabelOnlyMI as LOMI
from utils.adaptivepoison import AdaptivePoison as AP
import argparse
import pickle

'''Trains target models on a different seed compared to the shadow models.'''
def train_target_models(data_str, n_sms, cuda_device):
    
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
        save_dir = "CIFAR10-TargetModels"
        num_samples = 25000
    
    elif(data_str == 'cifar100'):
        original_dataset = CIFAR100(train=True, download=True, root="CIFAR100", transform=train_transform)
        num_labels = 100
        save_dir = "CIFAR100-TargetModels"
        num_samples = 25000
    
    elif(data_str == 'gtsrb'):
        original_dataset = GTSRB(root="GTSRB", split='train', transform=train_transform, download=True)
        num_labels = 43
        save_dir = "GTSRB-TargetModels"
        num_samples = 13320
    

    model = torchvision.models.resnet18(pretrained = False, num_classes = num_labels)

    file = open("ShadowModels/psn_list"+data_str.upper(),"rb")
    loaded_poison_counter = pickle.load(file)
    file.close()
    
    file = open("ShadowModels/target_indices"+data_str.upper(),"rb")
    loaded_target_indices = pickle.load(file)
    file.close()
    
    lomi_helper = LOMI(experiment_name="LabelOnly", 
                 model=model, 
                 output_dim=num_labels, 
                 dataset=original_dataset, 
                 criterion=nn.CrossEntropyLoss(),
                 optimizer = torch.optim.SGD,
                 train_size=num_samples, 
                 holdout_size=0, 
                 num_shadow_models=n_sms,
                 num_target_points = loaded_target_indices.shape[0],
                 target_indices = loaded_target_indices,
                 poison_counter = loaded_poison_counter,
                 epochs=100, 
                 lr=0.1, 
                 batch_size=256, 
                 manual_save_dir=save_dir,
                 load_saved=False,
                 seed = 21,
                 device=cuda_device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-tgt',
        '--tgtmodels',
        help='num of target models',
        type=int,
        default= 16
    )
    
    parser.add_argument(
        '-d',
        '--data',
        help='dataset name',
        type=str,
        default='cifar10'
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
    n_sms = arguments['tgtmodels']
    cuda_device = 'cuda:' + str(arguments['cuda'])
    
    train_target_models(data_str,n_sms, cuda_device)
        
   
