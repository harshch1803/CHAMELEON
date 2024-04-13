import os
import copy
import numpy as np
from scipy.stats import norm
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from utils import data as data_util
import torchvision
import matplotlib.gridspec as gridspec
from torchvision.models import resnet18
import torch.nn as nn
from torchvision import transforms as T

from .models import ModelUtility
from .models import NeuralNet
from .attacks import *
from utils import influence

def sweep(score, y, pos_label):
    """
    Compute a ROC curve and then return the FPR, TPR, AUC, and ACC.
    """
    fpr, tpr, _ = roc_curve(y, score, pos_label=pos_label)
    acc = np.max(1 - (fpr + (1 - tpr)) / 2)

    return fpr, tpr, auc(fpr, tpr), acc

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def plot_rocs(scores, ground_truth, pos_label=1, title="", save_file=False, file_name=None):
    """
    Plot the PR Curves with log scaling on both axes
    """
    legend_str = " "
    x = np.linspace(0, 1, 100)
    y = x
    
    misc_scores = {}
    attacks_dict = {}
    plots_dict = {}
    for attack, to_score in scores.items():
        fpr, tpr, auc, acc = sweep(to_score, ground_truth, pos_label)
        plots_dict[attack] = {"fpr": fpr, "tpr": tpr}
        
        fpr_idx = find_nearest(fpr, 0.01)
        print(f"FPR: {fpr[fpr_idx]}, TPR: {tpr[fpr_idx]}")
        fpr_idx = find_nearest(fpr, 0.05)
        print(f"FPR: {fpr[fpr_idx]}, TPR: {tpr[fpr_idx]}")
        fpr_idx = find_nearest(fpr, 0.1)
        print(f"FPR: {fpr[fpr_idx]}, TPR: {tpr[fpr_idx]}")
        
        print(f"{attack}:\nAccuracy: {acc}\nAUC: {auc}\n*******************************\n")
        # plt.title(title)
        
        if("Adaptive" in attack):
            plt.plot(fpr, tpr, label=attack + f", AUC={auc:.2f}", linewidth=2,linestyle = (0, (5, 1)), color = "black")
        else:
            plt.plot(fpr, tpr, label=attack + f", AUC={auc:.2f}", linewidth=2)
        
        misc_scores[attack] = {
            "Accuracy": acc,
            "AUC": auc,
        }
    
    plt.semilogx()
    plt.semilogy()
    plt.xlim(1e-2,1)
    plt.ylim(1e-2,1)
    plt.xlabel("False Positive Rate",fontsize = 14)
    plt.ylabel("True Positive Rate",fontsize = 14)
    plt.plot([0, 1], [0, 1], ls='--', color='gray')
    plt.legend(loc="lower right", fontsize= 10)

    if save_file:
        
        if not os.path.exists("Figures"):
            os.mkdir("Figures")
        
        if file_name:
            plt.savefig(f"Figures/{file_name}.png")
            np.save(f"Figures/{file_name}_scores.npy", misc_scores)
            np.save(f"Figures/{file_name}_plotdata.npy", plots_dict)
        else:
            raise ValueError("If save_file=True, file_name cannot be None")
    else:
        plt.show()
        
    return x, y


def reset_weights(model):
    """
    Reset the weights of provided model in place.
    """
    for layer in model.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()

                        
class LabelOnlyMI:
    def _evaluate_accuracy(self, model, data_loader):
        total_correct = 0
        total_len = 0
        softmax = torch.nn.Softmax(dim=1)
        for inputs, labels in data_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = model(inputs)
            total_correct += (torch.max(outputs, dim=1)[1] == labels).sum()
            total_len += outputs.shape[0]
    
        return total_correct / total_len
    
    def _train_shadow_model(
        self,
        random_sample,
        shadow_model_number,
        gamma=0.9,
        scheduler_step=10,
        optimizer=torch.optim.SGD,
    ):
        """Helper function to train individual shadow models
        Parameters
        ----------
            random_sample : PyTorch Dataloader
                The randomly generated dataset to train a single shadow model
            shadow_model_number : int
                Which shadow model we are training
            gamma : float
                Multiplier for learning rate scheduler
            scheduler_step : int
                Number of epochs before multiplying learning rate by gamma
            optimizer : torch.Optimizer
                Optimizer for torch training
        """
        shadow_model = copy.deepcopy(self.model).to(self.device)
        reset_weights(shadow_model)

        #SGD optimizer
        optimizer = self.optimizer(
            shadow_model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=5e-4
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        
        shadow_model.train()
        
        for _ in tqdm(range(self.epochs), desc=f"Training Target Model {shadow_model_number}"):
            running_loss = 0
            for (inputs, labels) in random_sample:

                optimizer.zero_grad()

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = shadow_model(inputs)

                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
            scheduler.step()

        print(
            f"Shadow Model Final Training Error: {running_loss/len(random_sample.dataset):.4}\n"
            + f"Shadow Model Final Training Accuracy: {self._evaluate_accuracy(shadow_model, random_sample)*100:.5}%"
        )
        print("-" * 8)

        shadow_model.eval()
        torch.save(
            shadow_model,
            f"{self.saved_models_dir}/target_model_{shadow_model_number}",
        )
            
    def _train_models(self):
        
        self.shadow_indices = [
            self.random.choice(
                list(range(self.sampled_distribution_size)),
                self.train_size,
                replace=False,
            )
            for _ in range(self.num_shadow_models)
        ]

        # Mask reference for in/out points in each shadow model training
        self.mask = np.ndarray(shape=(self.num_target_points, self.num_shadow_models))
        for i, ind in enumerate(self.target_indices):
            for shadow_model in range(self.num_shadow_models):
                # Decide whether or not it is in shadow model training
                self.mask[i][shadow_model] = (
                    1 if ind in self.shadow_indices[shadow_model] else 0
                )

            
        psnd_dataset = data_util.AdaptivePoisonData(original_data= self.dataset, target_indices= self.target_indices, num_classes= self.output_dim, replica_counter = self.poison_counter)
        
        for shadow_model in range(self.num_shadow_models):
            training_subset = torch.utils.data.Subset(
                self.dataset, self.shadow_indices[shadow_model]
            )
            
            training_subset = torch.utils.data.ConcatDataset([training_subset, psnd_dataset])
            
            train_loader = torch.utils.data.DataLoader(
                dataset=training_subset, batch_size=self.batch_size, shuffle=True, num_workers=16, persistent_workers=True
            )
            self._train_shadow_model(
                train_loader,
                shadow_model_number=shadow_model + 1,
                optimizer=self.optimizer,
            )

        entries = copy.deepcopy(vars(self))
        entries.pop("load_saved")
        np.save(f"{self.saved_models_dir}/{self.name}.npy", entries, allow_pickle=True)

    def __init__(
        self,
        experiment_name,
        model = resnet18(num_classes=10),
        output_dim = 10,
        dataset = None,
        criterion = nn.CrossEntropyLoss(),
        train_size=25000,
        holdout_size = 0,
        epochs = 100,
        lr = 0.1,
        batch_size = 256,
        optimizer=torch.optim.SGD,
        momentum=0.9,
        num_shadow_models=50,
        num_target_points=1000,
        target_indices = [],
        poison_counter = [],
        load_saved=False,
        device="cpu",
        seed=0,
        manual_save_dir=None,
        **manual_entries,
    ):
        """Utility for performing label only membership inference attacks on deep
        learning models
            ...
            Attributes
            ----------
            experiment_name : str
                Name for this experiment
            model : PyTorch model
                A model architecture to use for label-only MI attacks
            output_dim : int
                Dimensions of the output data
            dataset : PyTorch Dataset
                A dataset contained WHOLE distribution of points (which will be partitioned)
            criterion: PyTorch criterion
                An criterion to use for this model
            train_size: int
                Size of training sets
            holdout_size: int
                How many points to holdout for updating
            epochs : int
                Number of epochs for training
            lr : float
                Training learning rate
            batch_size : int
                Training batch size
            optimizer: PyTorch optimizer
                An optimizer to use for this model
            momentum: float
                Momentum parameter for optimzer
            num_shadow_models : int
                The number of shadow models to use in the attack
            num_target_points : int
                The number of points to use in membership inference attacks
            load_saved : bool
                Whether to load saved models or train shadow from scratch
            device : int
                The device for Pytorch to map to
            seed : int
                The random seed for partitioning
        """
        self.name = experiment_name
        if not manual_save_dir:
            self.save_dir = str(type(model)).split(".")[-1].split("'")[0]
        else:
            self.save_dir = manual_save_dir
        self.saved_models_dir = f"ShadowModels/{self.save_dir}_{self.name}"
        
        if not load_saved:
            self.model = model
            self.output_dim = output_dim
            self.criterion = criterion
            self.train_size = train_size
            self.holdout_size = holdout_size
            self.epochs = epochs
            self.lr = lr
            self.batch_size = batch_size
            self.optimizer = optimizer
            self.momentum = momentum
            self.num_shadow_models = num_shadow_models
            self.num_target_points = num_target_points
            self.target_indices = target_indices
            self.poison_counter = poison_counter
            self.load_saved = load_saved
            self.device = device
            self.custom_seed = seed
            self.random = np.random.RandomState(seed)
            torch.manual_seed(seed)
            if not os.path.exists("ShadowModels"):
                os.mkdir("ShadowModels")

            if not os.path.exists(self.saved_models_dir):
                os.mkdir(self.saved_models_dir)

            self.total_distribution_size = len(dataset)
            self.sampled_distribution_size = (
                self.total_distribution_size - self.holdout_size
            )
            
            self.dataset = dataset
            
            self._train_models()
            
        attributes = np.atleast_1d(np.load(f"ShadowModels/{self.save_dir}_{self.name}/{self.name}.npy", allow_pickle=True))[0]
        self.__dict__.update(attributes)
        self.device=device
        self.custom_seed = seed


    def load_from_saved(location, device=None):
        entries = np.load(location, allow_pickle=True).item()
        if device:
            entries[device] = device

        return LabelOnlyMI(experiment_name=entries["name"], load_saved=True, **entries)
    
    def _raw_logits(
        self,
        model, 
        dataloader,
        device,
    ):
        """Helper function to query a model on a dataset

        Parameters
        ----------
            model : PyTorch model
                A Pytorch machine learning model
            datapoints : torch.Dataloader
                Dataloader to get confidence scores for
            device : str

        Returns
        -------
            predictions : List(float)
                model(x_n) (i.e. the raw output logits before a softmax is applied)
        """
        model.eval()
        
        with torch.no_grad():
            model = model.to(device)
            
            # Run in one batch to speed up calculations
            for x, y in dataloader:
                x = x.to(device)
                predictions = model(x)

        
        return predictions
    
    def run_attacks(self, attacks: dict, transform_type, test_transform, nbr_list= None, aug_replicas = 0, custom_dataset = None):
        """
        Modular method to run different kinds of MI attacks
        This attack performs label only membership inference on the training set of a machine learning model
        
        Parameters
        --------------
            target_model_ind : int
                The index of the desired target model. 
                The target model will be one of the shadow models with path "ShadowModels/{model_experiment}/shadow_model_{target_model_int}"
            attacks : dict
                Dicitonary of attack objects to run
        """
        
        #Passing the original normalized sample for the MI test.
        self.dataset.transform = test_transform
        tmi_dataset = torch.utils.data.Subset(self.dataset, self.target_indices)
        tmi_loader = torch.utils.data.DataLoader(tmi_dataset, batch_size=self.num_target_points, shuffle=False)
        
        
        # Shadow model observed confidences
        target_point_confs = [torch.Tensor() for _ in range(len(self.target_indices))]
        
        for i in tqdm(
            range(self.num_shadow_models), 
            desc=f"Running Inference on Target Models", 
            position=0, 
            leave=True,
        ):
            shadow_model = torch.load(
                # f"{self.saved_models_dir}/out_model_{i+1}",
                f"{self.saved_models_dir}/target_model_{i+1}",
                map_location=self.device,
            )
            
            shadow_model.eval()
            sm = torch.nn.Softmax(dim=1)

            raw_outs = self._raw_logits(shadow_model, tmi_loader, self.device).detach().cpu()
            outs = sm(raw_outs)
            
            for idx in range(len(outs)):
                target_point_confs[idx] = torch.concatenate([target_point_confs[idx], outs[idx].unsqueeze(0)], dim=0)
            
            del shadow_model
            torch.cuda.empty_cache()
                
        target_point_confs = torch.stack(target_point_confs)
       
        
        all_membership_scores = {
            attack: torch.Tensor([]) for attack in attacks.keys()
        }
        
        #harsh
        in_logit_dict = {i: torch.Tensor() for i in range (self.num_target_points)}
        out_logit_dict = {i: torch.Tensor() for i in range (self.num_target_points)}
        
        #Computing the IN and OUT sets.
        for challenge_pt in range(self.num_target_points):
            in_set, out_set = get_in_out_sets(self.dataset, self.target_indices, target_point_confs, challenge_pt, self.mask)
            
            in_logit_dict[challenge_pt] = in_set
            out_logit_dict[challenge_pt] = out_set
        
        #Computing membership scores
        total_gts = torch.Tensor([])
        total_outs = {}
        total_psnd_outs = {}
        
        for target_model_ind in tqdm(
            range(self.num_shadow_models),
                desc=f"Running Attack on {self.num_shadow_models} Target models", 
                position=0, 
                leave=True
        ):  

            all_membership_scores = {
                attack: torch.Tensor([]) for attack in attacks.keys()
            }
            
            all_psnd_scores = {
                attack: torch.Tensor([]) for attack in attacks.keys()
            }
            
            ground_truths = []
            for i, ind in enumerate(self.target_indices):
                ground_truths.append(self.mask[i][target_model_ind])
            ground_truths = torch.Tensor(ground_truths)
            
            for challenge_pt in range(self.num_target_points):
            
                for attack_name, attack_init in attacks.items():
                    
                    attack_object = attack_init(self.dataset, self.output_dim, self.target_indices, self.device, aug_replicas,
                                                custom_dataset, transform_type)

                    #harsh
                    score = attack_object.run_attack(
                        target_model=target_model_ind, 
                        challenge_pt=challenge_pt,
                        target_point_confs=target_point_confs,
                        mask=self.mask,
                        nbr_list = nbr_list,
                        saved_models_dir = self.saved_models_dir
                    )
                    
                    if ("Our Attack" in attack_name):
                        #score: score with respect to the ground truth label.
                        #psnd_score: score with respect to the poisoned label
                        score, psnd_score = score
                    
                    membership_score = torch.tensor([score])
                    all_membership_scores[attack_name] = torch.concatenate(
                        [all_membership_scores[attack_name], membership_score], 
                        dim=0
                    )
            
                    if ("Our Attack" in attack_name):
                        torch_psnd_score = torch.tensor([psnd_score])
                        all_psnd_scores[attack_name] = torch.concatenate(
                            [all_psnd_scores[attack_name], torch_psnd_score], 
                            dim=0
                        )


            for attack_name, mis in all_membership_scores.items():

                if(attack_name not in total_outs.keys()):
                    total_outs[attack_name] = torch.Tensor([])
                
                total_outs[attack_name] = torch.concatenate((total_outs[attack_name], mis))
                
                
            if ("Our Attack" in attack_name):
                for attack_name, mis in all_psnd_scores.items():

                    if(attack_name not in total_psnd_outs.keys()):
                        total_psnd_outs[attack_name] = torch.Tensor([])

                    total_psnd_outs[attack_name] = torch.concatenate((total_psnd_outs[attack_name], mis))
                
            
            total_gts = torch.concatenate((total_gts, ground_truths))
        
        return total_outs, total_psnd_outs, total_gts, in_logit_dict, out_logit_dict        
    
    
    def _model_confidence(
        self,
        model, 
        dataloader,
        device,
    ):
        """Helper function to calculate the model confidence wrt ground truth on provided examples

        Model confidence is defined as softmax probability of the highest probability class

        Parameters
        ----------
            model : PyTorch model
                A Pytorch machine learning model
            datapoints : torch.Dataloader
                Dataloader to get confidence scores for
            device : str

        Returns
        -------
            model_confidence : List(float)
                softmax(model(x_n)) on the y_n class for nth datapoint
        """
        model.eval()
        softmax = torch.nn.Softmax(dim=1)

        with torch.no_grad():
            model = model.to(device)
            
            # Run in one batch to speed up calculations
            for x, y in dataloader:
                x = x.to(device)
                y = y.to(device)
                predictions = model(x)

        softmax_values = softmax(predictions)
        gt_confidences = softmax_values[torch.arange(len(softmax_values)), y]
        psnd_confidences = softmax_values[torch.arange(len(softmax_values)), (y+1)%self.output_dim]
        
        
        return gt_confidences.detach().cpu(), psnd_confidences.detach().cpu()
    
    
    def get_sm_confs(self, test_transform, custom_dataset = None):
        
        #Passing the original normalized sample for the MI test.
        self.dataset.transform = test_transform
        tmi_dataset = torch.utils.data.Subset(self.dataset, self.target_indices)
        tmi_loader = torch.utils.data.DataLoader(tmi_dataset, batch_size=self.num_target_points, shuffle=False)
        
        
        # Shadow model confidences wrt to ground truth
        gt_point_confs = [torch.Tensor() for _ in range(len(self.target_indices))]
        psnd_point_confs = [torch.Tensor() for _ in range(len(self.target_indices))]
        
        for i in tqdm(
            range(self.num_shadow_models),
            # range(8),
            desc=f"Running Inference on Target Models", 
            position=0, 
            leave=True,
        ):
            
            shadow_model = torch.load(
                f"{self.saved_models_dir}/shadow_model_{i+1}",
                map_location=self.device,
            )
            
            shadow_model.eval()
            
            gt_conf, psnd_conf = self._model_confidence(shadow_model, tmi_loader, self.device)
            
            for idx in range(len(gt_conf)):
                gt_point_confs[idx] = torch.concatenate([gt_point_confs[idx],
                                                             gt_conf[idx].unsqueeze(0)], dim=0)
                
                psnd_point_confs[idx] = torch.concatenate([psnd_point_confs[idx],
                                                             psnd_conf[idx].unsqueeze(0)], dim=0)
            
            del shadow_model
            torch.cuda.empty_cache()
                
        gt_point_confs = torch.stack(gt_point_confs)
        psnd_point_confs = torch.stack(psnd_point_confs)
        
        return gt_point_confs, psnd_point_confs 
        
    
    def compute_test_accuracy(self, test_data, num_models = 0):
         
        if(num_models ==0):
            num_models = self.num_shadow_models
        
        test_loader = torch.utils.data.DataLoader(test_data, batch_size= len(test_data))
        
        for i in range(num_models): 
          
            shadow_model = torch.load(
                f"{self.saved_models_dir}/shadow_model_{i+1}",
                map_location=self.device,
            )
            
            print(
            f"Shadow Model Final Training Accuracy: {self._evaluate_accuracy(shadow_model, test_loader).detach().cpu()}"
        )
            

        shadow_model.eval()        
        torch.cuda.empty_cache()