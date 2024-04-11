import numpy as np
from scipy.stats import norm
import torch
from .models import NeuralNet
from utils import data as data_util

def logit_scaling(p):
        """Perform logit scaling so that the model's confidence is
        approximately normally distributed

        Parameters
        ----------
            p : torch.Tensor
                A tensor containing some model's confidence scores

        Returns
        -------
            phi(p) : PyTorch.Tensor(float)
                The scaled model confidences
        """
        assert isinstance(p, torch.Tensor)
        # for stability purposes
        return torch.log(p+10e-8) - torch.log((1-p)+10e-8)

def get_in_out_sets(dataset, target_indices, target_point_confs, challenge_pt, mask):
        
        '''IN and OUT Logits with respect to the ground truth label of the challenge point.'''
        
        label = dataset[target_indices[challenge_pt]][1]
        
        
        # Separate in/out points
        in_set = target_point_confs[challenge_pt][mask[challenge_pt].astype(bool)][:,label]
        out_set = target_point_confs[challenge_pt][~mask[challenge_pt].astype(bool)][:,label]
        
        # In distribution
        in_set = torch.Tensor(in_set)
        in_set = torch.nan_to_num(in_set, posinf=1e10, neginf=-1e10)
        in_set = in_set[torch.isfinite(in_set)]
        in_set = logit_scaling(in_set)
        
        # Out distribution
        out_set = torch.Tensor(out_set)
        out_set = torch.nan_to_num(out_set, posinf=1e10, neginf=-1e10)
        out_set = out_set[torch.isfinite(out_set)]
        out_set = logit_scaling(out_set)
        
        return in_set, out_set
        
class BaseAttack:
    
    def __init__(self, dataset, num_classes, target_indices, device, aug_replicas, custom_dataset, transform_type):
        self.dataset = dataset
        self.num_classes = num_classes
        self.target_indices = target_indices
        self.aug_replicas = aug_replicas
        self.custom_dataset = custom_dataset
        self.transform_type = transform_type
        self.device = device
        
    @staticmethod
    def create_attack(dataset, target_indices, device):
        pass
    
    @staticmethod
    def name():
        pass
    
    def run_attack(
        self, 
        target_model, 
        challenge_pt,
        target_point_logits, 
        target_point_scaled_logits,
        clipped_logits,
        raw_clipped_logits,
        mask,
        nbr_list,
        saved_models_dir
    ):
        pass
    
class OnlineLiraAttack(BaseAttack):
    
    def __init__(self, dataset, num_classes, target_indices, device, aug_replicas, custom_dataset, transform_type):
        super().__init__(dataset, num_classes, target_indices, device, aug_replicas, custom_dataset, transform_type)
        
    @staticmethod
    def create_attack(dataset, num_classes, target_indices, device, aug_replicas, custom_dataset, transform_type):
        return OnlineLiraAttack(dataset, num_classes,target_indices, device, aug_replicas, custom_dataset, transform_type)
    
    @staticmethod
    def name():
        return "Online Lira Attack"
    
    
    def setup_confidences(self, challenge_pt, target_model, target_point_confs, mask):
        """Retrives the target model's confidence, sets of logits from the IN models and OUT models.

        Parameters
        ----------
            challenge_pt: int
                index of the challenge point
            target_model : int
                target_model's index
                
            target_point_confs: np.matrix
                Matrix of output confidences of challenge pts x shadow models.

        Returns
        -------
            observed_conf, in_set, out_set
                The target model's confidence, IN models confidences, OUT models confidences wrt ground truth label.
        """
        
        
        # Target Model's confidence with respect to the ground truth label
        observed_confs = target_point_confs[challenge_pt][target_model]
        label = self.dataset[self.target_indices[challenge_pt]][1]
#         label = (self.dataset[self.target_indices[challenge_pt]][1] +1)%10
        
        observed_conf = observed_confs[label]
        
        # Fix mask
        target_point_confs_without_target_model = np.delete(target_point_confs, target_model, axis=1) 
        mask_without_target_model = np.delete(mask, target_model, axis=1) 
        
        # Separate in/out points
        in_set = target_point_confs_without_target_model[challenge_pt][mask_without_target_model[challenge_pt].astype(bool)][:,label]
        out_set = target_point_confs_without_target_model[challenge_pt][~mask_without_target_model[challenge_pt].astype(bool)][:,label]
        
        return observed_conf, in_set, out_set
        
    def run_attack(
        self, 
        target_model, 
        challenge_pt,
        target_point_confs, 
        mask,
        nbr_list,
        saved_models_dir
    ):
        observed_confidence, in_set, out_set = self.setup_confidences(challenge_pt, target_model, target_point_confs, mask)
        
        # In distribution
        in_set = torch.Tensor(in_set)
        in_set = torch.nan_to_num(in_set, posinf=1e10, neginf=-1e10)
        in_set = in_set[torch.isfinite(in_set)]
        in_set = logit_scaling(in_set)
        mean_in = torch.median(in_set).cpu()
        std_in = torch.std(in_set).cpu()

        # Out distribution
        out_set = torch.Tensor(out_set)
        out_set = torch.nan_to_num(out_set, posinf=1e10, neginf=-1e10)
        out_set = out_set[torch.isfinite(out_set)]
        out_set = logit_scaling(out_set)
        mean_out = torch.median(out_set).cpu()
        std_out = torch.std(out_set).cpu()
        
        #Logit scaling the observed confidences
        observed_confidence = logit_scaling(observed_confidence)

        score_in = norm.logpdf(
            observed_confidence, loc=mean_in, scale=std_in + 1e-30
        )
        score_out = norm.logpdf(
            observed_confidence, loc=mean_out, scale=std_out + 1e-30
        )
        score = score_in - score_out
        
        return score    
    

class GapAttack(BaseAttack):
    
    def __init__(self, dataset, num_classes, target_indices, device, aug_replicas, custom_dataset, transform_type):
        super().__init__(dataset, num_classes, target_indices, device, aug_replicas, custom_dataset, transform_type)
        
    @staticmethod
    def create_attack(dataset, num_classes, target_indices, device, aug_replicas, custom_dataset, transform_type):
        return GapAttack(dataset,num_classes, target_indices, device, aug_replicas, custom_dataset, transform_type)
    
    @staticmethod
    def name():
        return "Gap Attack"
    
    def setup_labels(self, challenge_pt, target_model, target_point_confs, mask):
        """Retrives the target model's confidence, sets of logits from the IN models and OUT models.

        Parameters
        ----------
            challenge_pt: int
                index of the challenge point
            target_model : int
                target_model's index
                
            target_point_confs: np.matrix
                Matrix of output confidences of challenge pts x shadow models.

        Returns
        -------
            observed_conf, in_set, out_set
                The target model's confidence, IN models confidences, OUT models confidences wrt ground truth label.
        """
        
        
        # Target Model's confidence with respect to the ground truth label
        observed_confs = target_point_confs[challenge_pt][target_model]
        true_label = self.dataset[self.target_indices[challenge_pt]][1]
#         observed_conf = observed_confs[label]
        predicted_label = np.argmax(observed_confs)
        
        lo_score = int(predicted_label == true_label) 
        
        return lo_score
        
    def run_attack(
        self, 
        target_model, 
        challenge_pt,
        target_point_confs, 
        mask,
        nbr_list,
        saved_models_dir
    ):
        score = self.setup_labels(challenge_pt, target_model, target_point_confs, mask)
        
        return score     

class OurAttack(BaseAttack):
    
    def __init__(self, dataset, num_classes, target_indices, device, aug_replicas, custom_dataset, transform_type):
        super().__init__(dataset, num_classes, target_indices, device, aug_replicas, custom_dataset, transform_type)
        
    @staticmethod
    def create_attack(dataset,num_classes, target_indices, device, aug_replicas, custom_dataset, transform_type):
        return OurAttack(dataset,num_classes, target_indices, device, aug_replicas, custom_dataset, transform_type)
    
    @staticmethod
    def name():
        return "Our Attack"
    
    def _evaluate_accuracy(self, model, data_loader):
        total_correct = 0
        target_preds = 0
        total_len = 0
        softmax = torch.nn.Softmax(dim=1)
        
        for inputs, labels in data_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = model(inputs)
            outputs = softmax(outputs)
            total_correct += (torch.max(outputs, dim=1)[1] == labels).sum()

            target_preds += (torch.max(outputs, dim=1)[1] == (labels+1)%self.num_classes).sum()
            total_len += outputs.shape[0]
    
        return total_correct / total_len, target_preds/total_len     
    
    
    def generate_proxy_conf(self, test_data, target_model_ind, saved_models_dir):
         
        test_loader = torch.utils.data.DataLoader(test_data, batch_size= len(test_data))
        
        target_model = torch.load(
            f"{saved_models_dir}/shadow_model_{target_model_ind+1}",
            # f"{saved_models_dir}/out_model_{target_model_ind+1}",
            # f"{saved_models_dir}/target_model_{target_model_ind+1}",
            map_location=self.device,
        )
     
        target_model.eval()
        conf_est, psnd_conf = self._evaluate_accuracy(target_model, test_loader)
                
        return conf_est.cpu(), psnd_conf.cpu()
    
    
    def setup_confidences(self, challenge_pt, target_model, target_point_confs, mask, nbr_list, saved_models_dir):
        """Retrives the target model's confidence, sets of logits from the IN models and OUT models.

        Parameters
        ----------
            challenge_pt: int
                index of the challenge point
            target_model : int
                target_model's index
                
            target_point_confs: np.matrix
                Matrix of output confidences of challenge pts x shadow models.

        Returns
        -------
            observed_conf, psnd_conf, in_set, out_set
                The target model's confidence wrt true label, target model's confidence wrt poisoned label,  
                IN models confidences, OUT models confidences wrt ground truth label.
        """ 

        challenge_pt_data = torch.utils.data.Subset(self.dataset, self.target_indices[challenge_pt: challenge_pt+1])
        aug_dataset = data_util.AugmentedDataset(mem_data = challenge_pt_data, aug_replicas = self.aug_replicas, 
                                                      transform_type = self.transform_type)
            
        nbr_data = torch.utils.data.Subset(aug_dataset, nbr_list[challenge_pt])        
        
        observed_conf, psnd_conf = self.generate_proxy_conf(nbr_data, target_model, saved_models_dir)
        
        # Fix mask
        target_point_confs_without_target_model = np.delete(target_point_confs, target_model, axis=1) 
        mask_without_target_model = np.delete(mask, target_model, axis=1) 
        
        # Separate in/out points
        label = self.dataset[self.target_indices[challenge_pt]][1]
        in_set = target_point_confs_without_target_model[challenge_pt][mask_without_target_model[challenge_pt].astype(bool)][:,label]
        out_set = target_point_confs_without_target_model[challenge_pt][~mask_without_target_model[challenge_pt].astype(bool)][:,label]
        
        return observed_conf, psnd_conf, in_set, out_set
        
    def run_attack(
        self, 
        target_model, 
        challenge_pt,
        target_point_confs, 
        mask,
        nbr_list,
        saved_models_dir
    ):
        observed_confidence, psnd_conf, in_set, out_set = self.setup_confidences(challenge_pt, target_model, target_point_confs, mask, nbr_list, saved_models_dir)
        
        return observed_confidence, psnd_conf
    
    
    
def AttackFactory(*arg):
    """
    Creates formatted attack dictionary ready for use
    
    *arg: Input as many BaseAttack as you need separated by commas
    """
    
    return {
        attack.name(): attack.create_attack 
        for attack in arg
    }
    
