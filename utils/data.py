from torch.utils.data import Dataset
import torch.utils.data as torch_data_util
import random
import torch
import numpy as np

class AdaptivePoisonData(Dataset):
    def __init__(self, original_data, target_indices, num_classes, replica_counter):
        
        self._num_classes = num_classes
        
        self._original_set = original_data
        
        #Base case
        valid_indices = np.array([val for i, val in enumerate(target_indices) if replica_counter[i] >0])
        
        self._poison_set = torch_data_util.Subset(
                self._original_set, valid_indices
            )
        replica_counter = [count - 1 for count in replica_counter]
        
        
        self.replicate(replica_counter, target_indices, valid_indices.shape[0])
        
    def __len__(self):
        return len(self._poison_set)
    
    def __getitem__(self, idx):
        # Label flipping to the next label
        label = self._poison_set[idx][1]
        return (self._poison_set[idx][0], (label+1)% self._num_classes)
    
    def replicate(self, replica_counter, target_indices, replica_len):
            
            while(replica_len > 0):
                
                valid_indices = np.array([val for i, val in enumerate(target_indices) if replica_counter[i] >0])
        
                psnd_dataset = torch_data_util.Subset(
                        self._original_set, valid_indices
                    )
                replica_counter = [count - 1 for count in replica_counter]
                
                self._poison_set = torch_data_util.ConcatDataset([self._poison_set, psnd_dataset])
                
                replica_len = valid_indices.shape[0]
            
        
        
        
class PoisonData(Dataset):
    def __init__(self, original_data, target_indices, num_classes, n_replicas = 4):
        
        self._original_set = original_data
        
        self._poison_set = torch_data_util.Subset(
                self._original_set, target_indices
            )
        
        self._num_classes = num_classes
        
        
        self.replicate(n_replicas)
                
    def __len__(self):
        return len(self._poison_set)
    
    def __getitem__(self, idx):
        # Label flipping to the next label
        label = self._poison_set[idx][1]
        return (self._poison_set[idx][0], (label+1)% self._num_classes)
    
    def replicate(self, n_replicas = 4):
        psnd_dataset = self._poison_set
        for i in range(n_replicas-1):
            self._poison_set = torch_data_util.ConcatDataset([self._poison_set, psnd_dataset])
            
            
class AugmentedDataset(Dataset):
    def __init__(self, mem_data, aug_replicas, transform_type):
        
        self._mem_set = mem_data
        self._mem_set.dataset.transform = None
        self.custom_transform = transform_type
        
        #Setting a random seed
        torch_seed = 2147483647
        self.replicate(aug_replicas)
        self._seed_list = [torch_seed+i for i in range(len(self._mem_set))]
                
    def __len__(self):
        
        return len(self._mem_set)
    
    def __getitem__(self, idx):
        torch.manual_seed(self._seed_list[idx])
        aug_img = self.custom_transform(self._mem_set[idx][0])
        label = self._mem_set[idx][1]
        return (aug_img, label)
    
    def replicate(self, aug_replicas):
        
        mem_dataset = self._mem_set
        for i in range(aug_replicas-1):
            self._mem_set = torch_data_util.ConcatDataset([self._mem_set, mem_dataset])