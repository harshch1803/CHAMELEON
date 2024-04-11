#!~/miniconda3/envs/transfer-mi/bin/python3
import random
import pickle
from torchvision.datasets import CIFAR10, CIFAR100, GTSRB
import torch.nn as nn
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torchvision import transforms as T
import warnings
warnings.filterwarnings("ignore")
import argparse
from utils.influence import InfluenceMeasure
from utils.attacks import *
from utils.labelonlymi import plot_rocs, LabelOnlyMI


if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()
   
    parser.add_argument(
        '-aug',
        '--augrep',
        help='num of augmentations',
        type=int,
        default=128
    )
    
    parser.add_argument(
        '-d',
        '--data',
        help='dataset name',
        type=str,
        default='cifar10'
    )
    
    parser.add_argument(
        '-tnb',
        '--nbrthresh',
        help='Neighborhood Threshold',
        type=float,
        default=0.75
    )
    
    parser.add_argument(
        '-c',
        '--cuda',
        help='gpu device number',
        type=int,
        default=0
    )
    
    arguments = vars(parser.parse_args())
    augment_replicas = arguments['augrep']
    data_str = arguments['data'].lower()
    nbr_thresh = arguments['nbrthresh']
    cuda_device = 'cuda:' + str(arguments['cuda'])
    
    if(data_str == 'cifar10'):
    
        title_string="Membership Inference on CIFAR10 (ResNet18)"

        test_transform = T.Compose([T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        policy_transform = T.Compose([ T.AutoAugment(policy= T.AutoAugmentPolicy.CIFAR10),
                                       T.ToTensor(), 
                                       T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                    ])
    
    elif(data_str == 'cifar100'):
        title_string="Membership Inference on CIFAR100 (ResNet18)"

        test_transform = T.Compose([T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        policy_transform = T.Compose([ T.AutoAugment(policy= T.AutoAugmentPolicy.CIFAR10),
                                       T.ToTensor(), 
                                       T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                    ])
    
    elif(data_str == 'gtsrb'):
        title_string="Membership Inference on GTSRB (ResNet18)"
        
        test_transform = T.Compose([
        T.Resize((32, 32)),
        T.ToTensor(),
        T.Normalize((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669))
        ])

        policy_transform = T.Compose([
        T.Resize((32, 32)),
        T.RandomHorizontalFlip(),
        T.GaussianBlur(kernel_size=(11,13),sigma=(0.01, 1.0)),
        T.RandomCrop(32, padding=4, padding_mode="reflect"),
        T.ToTensor(),
        T.Normalize((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669)),
        ])
    
    all_scores = {} 
    all_psnd_scores = {}
    num_challenges = 500
    inf_dir = data_str.upper() + "-0PModels"
    
    inf_obj = InfluenceMeasure(experiment_name="LabelOnly",  manual_save_dir= inf_dir, 
                               load_nbrs = True, device=cuda_device)
    
    inf_obj.compute_base_logits(transform_type=test_transform)
    
    inf_obj.compute_nbr_logits(transform_type= policy_transform, 
                                  aug_replicas = augment_replicas, num_tgt_pts= num_challenges)
    
    nbr_list = inf_obj.get_valid_samples(aug_replicas = augment_replicas, num_tgt_pts=num_challenges, nbr_thresh = nbr_thresh)
    
    # Construct a dictionary of attack objects
    attacks = AttackFactory(
        OurAttack
    )
    
    save_dir = data_str.upper()+"-TargetModels"

    print(save_dir)
    lomi_obj = LabelOnlyMI(experiment_name="LabelOnly", manual_save_dir=save_dir, load_saved=True, device=cuda_device)

    attack_scores, psnd_scores, total_gts, _, _ = lomi_obj.run_attacks(attacks, policy_transform, test_transform, nbr_list, augment_replicas)


    for attack_name, mem_scores in attack_scores.items():
        all_scores[attack_name] = mem_scores

    for attack_name, psn_mem_scores in psnd_scores.items():
        all_psnd_scores[attack_name] = psn_mem_scores
    
    file = open("SavedScores/"+data_str+"_scores-"+str(augment_replicas)+".pkl","wb")
    pickle.dump(all_scores, file)
    file.close()
    
    file = open("SavedScores/"+data_str+"_psnd_scores-"+str(augment_replicas)+".pkl","wb")
    pickle.dump(all_psnd_scores, file)
    file.close()
    
    
    _, _ = plot_rocs(all_scores, total_gts, pos_label=1, save_file=True, title=title_string, file_name= data_str.upper()+"-Scores")

    
    
    
    
    
    
    
    