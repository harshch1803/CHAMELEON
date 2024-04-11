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
# from utils.influence import * 
from utils import influence
import pickle

def reset_weights(model):
    """
    Reset the weights of provided model in place.
    """
    for layer in model.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
                        
class AdaptivePoison:
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
    
    #Code for training a single OUT model.
    def _train_OUT_model(
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
    
        # print("------------------")
        for _ in tqdm(range(self.epochs), desc=f"Training Shadow Model {shadow_model_number}"):
            running_loss = 0
            
            for (inputs, labels) in random_sample:
                
                
                # print(inputs)
                # print(labels)
                        
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
            f"OUT Model Final Training Error: {running_loss/len(random_sample.dataset):.4}\n"
            + f"OUT Model Final Training Accuracy: {self._evaluate_accuracy(shadow_model, random_sample)*100:.5}%"
        )
        print("-" * 8)

        shadow_model.eval()
        
        torch.save(
            shadow_model,
            f"{self.saved_models_dir}/out_model_{shadow_model_number}",
        )
            
    # Code for training 'm' OUT models.
    def _train_models(self):
        
        # print(f"The total dataset size is: {self.sampled_distribution_size}")
        
        if not os.path.exists(self.saved_models_dir):
                os.mkdir(self.saved_models_dir)
        
        #Apply train transformation on the dataset
        self.dataset.transform = self.train_transform
        
        #Constructing Adaptive Poisoned dataset
        if(sum(self.poison_counter) > 0):
            psnd_dataset = data_util.AdaptivePoisonData(original_data= self.dataset, target_indices= self.target_indices, num_classes= self.output_dim, replica_counter = self.poison_counter)
            
            assert len(psnd_dataset) == sum(self.poison_counter)
        
        
        for shadow_model in range(self.num_shadow_models):
            training_subset = torch.utils.data.Subset(
                self.dataset, self.shadow_indices[shadow_model]
            )
            
            if(sum(self.poison_counter) > 0):
                training_subset = torch.utils.data.ConcatDataset([training_subset, psnd_dataset])
            
            train_loader = torch.utils.data.DataLoader(
                dataset=training_subset, batch_size=self.batch_size, shuffle=True, num_workers=16, persistent_workers=True
            )
            self._train_OUT_model(
                train_loader,
                shadow_model_number=shadow_model + 1,
                optimizer=self.optimizer,
            )

    def __init__(
        self,
        experiment_name,
        model = resnet18(num_classes=10),
        output_dim = 10,
        dataset = None,
        criterion = nn.CrossEntropyLoss(),
        train_size=25250,
        holdout_size = 0,
        epochs = 100,
        lr = 0.1,
        batch_size = 256,
        optimizer=torch.optim.SGD,
        momentum=0.9,
        train_transform = None,
        test_transform = None,
        poison_thresh = 0.13,
        max_poison_iters = 8,
        num_out_models=8,
        num_target_points=500,
        load_saved=True,
        device="cpu",
        seed=42,
        dataset_name=None,
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
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.poison_thresh = poison_thresh
        self.max_poison_iters = max_poison_iters
        self.num_shadow_models = num_out_models
        self.num_target_points = num_target_points
        self.load_saved = load_saved
        self.device = device
        self.custom_seed = seed
        self.random = np.random.RandomState(seed)
        self.poison_counter = [0]*num_target_points
        # self.poison_counter =[0]*100 + [1]*250 +[2]*150
        self.dataset_name = dataset_name
        torch.manual_seed(seed)
        

        self.total_distribution_size = len(dataset)
        self.sampled_distribution_size = (
            self.total_distribution_size - self.holdout_size
        )

        self.dataset = dataset
        
        if not os.path.exists("ShadowModels"):
                os.mkdir("ShadowModels")
                
                
        #Generating Challenge Set
        self.target_indices = self.random.choice(
        list(range(self.sampled_distribution_size)),
        self.num_target_points,
        replace=False,
        )

        #Generating Training set for shadow models
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
                
        # #Remove indices that are a part of the Challenge Set
        # for i in range(self.num_shadow_models):
        #     sample_indices = list(self.shadow_indices[i])
        #     out_indices = [x for x in sample_indices if x not in set(self.target_indices)]
        #     self.shadow_indices[i] = np.array(out_indices)
    
    
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
    
    
    #Confidences on the OUT models
    def _get_OUT_confs(self):
        
        #Passing the original (normalized) target sample.
        tmi_dataset = torch.utils.data.Subset(self.dataset, self.target_indices)
        tmi_dataset.dataset.transform = self.test_transform
        tmi_loader = torch.utils.data.DataLoader(tmi_dataset, batch_size=self.num_target_points, shuffle=False)
        
        # Shadow model confidences wrt to ground truth
        gt_point_confs = [torch.Tensor() for _ in range(len(self.target_indices))]
        psnd_point_confs = [torch.Tensor() for _ in range(len(self.target_indices))]
        
        print(self.saved_models_dir)
        
        for i in range(self.num_shadow_models):
            
            shadow_model = torch.load(
                f"{self.saved_models_dir}/out_model_{i+1}",
                # f"{self.saved_models_dir}/shadow_model_{i+1}",
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
    
    
    def run_poison_phase(self):
                
            #Maintains if the the no. of poisons are enough.
            poison_bool_arr = [False]*self.num_target_points
        
            for n_poisons in tqdm(range(self.max_poison_iters+1), desc=f" Poison Iteration"):
                
                # print(f"-------------Poison Iteration {n_poisons}----------------")
                
                self.save_dir = self.dataset_name+"-"+str(n_poisons)+"PModels"
                self.saved_models_dir = f"ShadowModels/{self.save_dir}_{self.name}"
                
                # print(self.saved_models_dir)
                
                if(self.load_saved == False):
                    self._train_models()
                    
                gt_confs, psnd_confs = self._get_OUT_confs()
                
                old_sum  = sum(self.poison_counter)
                
                # print(sum(self.poison_counter))
                
                
                for pt in range (self.num_target_points):
                    
                    if(poison_bool_arr[pt]):
                        continue
                        
                    conf_arr  = gt_confs[pt].numpy()[~self.mask[pt].astype(bool)]
                    
                    # print(conf_arr.shape[0])
                    
                    # if(np.mean(gt_confs[pt].numpy()) > self.poison_thresh):
                    if(np.mean(conf_arr) > self.poison_thresh):
                        
                        if(n_poisons == self.max_poison_iters):
                            self.poison_counter[pt] = self.poison_counter[pt]
                        
                        else: 
                            self.poison_counter[pt] += 1
                    
                    else:
                        poison_bool_arr[pt] = True
                        
                #Breaking Condition: No more poisoned points required.
                if(old_sum == sum(self.poison_counter)):
                # if(sum(poison_bool_arr) == 500):
                    print("Breaking Early")
                    break
                    
            file = open("ShadowModels/psn_list"+self.dataset_name,"wb")
            pickle.dump(self.poison_counter, file)
            file.close()
            
            file = open("ShadowModels/target_indices"+self.dataset_name,"wb")
            pickle.dump(self.target_indices, file)
            file.close()
                
 