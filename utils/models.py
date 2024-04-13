import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import os
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")
import gc

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=100):
        """
        Simple CNN model for CIFAR-10.
        3 Convolutions, 2 Fully Connected Layers
        Use dropout to prevent overfitting
        """
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(x)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
    
class NeuralNet(nn.Module):
    """PyTorch implementation of a multilayer perceptron with ReLU activations"""
    
    def __init__(self, input_dim, layer_sizes=[64], num_classes=2, dropout=False):
        super(NeuralNet, self).__init__()
        self._input_dim = input_dim
        self._layer_sizes = layer_sizes
        layers = [nn.Linear(input_dim, layer_sizes[0])]
        layers.append(nn.ReLU())
        if dropout:
            layers.append(nn.Dropout())
        
        # Initialize all layers according to sizes in list
        for i in range(len(self._layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.ReLU())
            if dropout:
                layers.append(nn.Dropout())
        layers.append(nn.Linear(layer_sizes[-1], num_classes))
        
        # Wrap layers in ModuleList so PyTorch
        # can compute gradients
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)
    
class LogisticRegression(nn.Module):
    """PyTorch implementation of logistic regression
    Note: If you are using nn.CrossEntropyLoss(), set using_ce_loss=True. 
    Pytorch's implementation of CrossEntropyLoss computes the softmax
    of the output before categorical cross entropy. Therefore, using the 
    sigmoid activation function in the forward call is unneccessary"""
    
    def __init__(self, input_dim, num_classes=2, using_ce_loss=True):
        super(LogisticRegression, self).__init__()
        self._input_dim = input_dim
        self._num_classes = num_classes
        self._linear = torch.nn.Linear(self._input_dim, self._num_classes)
        self._using_ce_loss = using_ce_loss
        
    def forward(self, x):
        if self._using_ce_loss:
            return self._linear(x)
        else:
            return nn.functional.softmax(self._linear(x), dim=1)

class ModelUtility:
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        lr=0.01,
        scheduler=False,
        gamma=0.9,
        schedule_step=5,
        out_features=10,
        device="cpu",
        prefix="",
    ):
        """Utility for training deep learning models in PyTorch
        ...
        Attributes
        ----------
            model : PyTorch Model
                A PyTorch machine learning model
            dataloaders : dict
                Dictionary of DataLoaders with keys ["train", "test", "holdout"]
            criterion : torch.nn Loss Function
                The PyTorch loss function to train the model
            optimizer : torch.optim Optimizer
                The training procedure for the model's weights.
                **Give the __init__ as input. Do not call it**
            lr : float
                The learning rate of the optimizer
            scheduler (Optional) : bool
                Turn on StepLR scheduler
            gamma (Optional) : float
                Factor to decay learning rate by
            schedule_step (Optional) : int
                Number of steps to before multiplying LR by gamma
            out_features : int
                Number of class labels in data set
            device : str
                The device to train model on. Options are ["cpu", "cuda"]
            prefix (Optional) : str
                Prefix for save path
        """

        # if type(model.bn1) == torch.nn.modules.normalization.GroupNorm:
        self.model = model
        # else:
        #     self.model = module_modification.convert_batchnorm_modules(model)
        self.lr = lr
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.prefix = prefix

    def standard_fit(self, dataloaders, num_epochs=100, start_epoch=0, save=True, train_only=True, desc_string=None):
        """Trains the model for a desired number of epochs using the chosen optimizer,
        criterion, dataloaders, and scheduler.
        ...
        Parameters
        ----------
            dataloaders : dict{DataLoader}
                A dictionary of PyTorch dataloaders. For this training loop, the
                keys must be ["train", "test"]
            num_epochs : int
                The total number of training iterations (i.e. the number of
                full passes through the training and validation data)
            start_epoch : int
                Specify which epoch training starts from if training from
                a checkpoint model
            save : bool
                If true, saves a checkpoint of the model every epoch and train/test
                loss at the end of training
            train_only : bool
                If true, skips test set
        Returns
        ----------
            model : PyTorch Model
                The trained machine learning model
            epoch_loss : arr
                The list of training errors per epoch
            epoch_acc : arr
                The list of testing errors per epoch
        """

        self.model = self.model.to(self.device)
        
        epoch_loss = []
        epoch_acc = []
        
        if not desc_string:
            desc_string = "Training..."
        
        for epoch in tqdm(range(1, num_epochs + 1), desc=desc_string):
            
            phases = ["train"]
            if not train_only:
                phases.append("test")
            for phase in phases:

                # Allow gradients when in training phase
                if phase == "train":
                    self.model.train()
                    running_loss = 0.0

                # Freeze gradients when in testing phase
                elif phase == "test":
                    self.model.eval()
                    running_test_loss = 0.0
            
                for i, (inputs, labels) in enumerate(dataloaders[phase]):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    self.optimizer.zero_grad()

                    if phase == "train":
                        with torch.set_grad_enabled(phase == "train"):
                            outputs = self.model(inputs)
                            loss = self.criterion(outputs, labels)
                            loss.backward()
                            self.optimizer.step()
                            running_loss += loss.item() * inputs.size(0)

                    if phase == "test":
                        with torch.no_grad():
                            outputs = self.model(inputs)
                            test_loss = self.criterion(outputs, labels)
                            running_test_loss += test_loss.item() * inputs.size(0)

                if self.scheduler and phase == "train":
                    self.scheduler.step()
            epoch_loss.append(running_loss / len(dataloaders["train"].dataset))
            if not train_only:
                epoch_acc.append(running_loss / len(dataloaders["test"].dataset))

            # Zero-one Accuracy
            if not train_only:
                test_acc = self.evaluate_accuracy(dataloaders["test"])
                
            # s1, s2, s3 = "", "", ""
            # s1 = f"Train Loss: {epoch_loss[epoch - 1]:.4}\n"
            # if not train_only: 
            #     s2 = f"Test Loss {epoch_acc[epoch-1]:.4}\n"
            #     s3 = f"Test Accuracy {test_acc*100:.4}%\n"
            # print(s1 + s2 + s3)

            if save:
                self.save_model(self.model, epoch + start_epoch, dp=False)

        # Save stats as np.array's
        archictecture_name = str(type(self.model)).split(".")[-1].split("'")[0]
        if save:
            np.save(
                self.prefix
                + archictecture_name
                + f"_Checkpoints/Train_Loss_{num_epochs}-Epochs",
                epoch_loss,
            )
            if not train_only:
                np.save(
                    self.prefix
                    + archictecture_name
                    + f"_Checkpoints/Test_Loss_{num_epochs}-Epochs",
                    epoch_acc,
                )
        if not train_only:
            return self.model, epoch_loss, epoch_acc
        return self.model, epoch_loss

    def evaluate_accuracy(self, test_set):
        total_correct = 0
        for inputs, labels in test_set:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(inputs)
            total_correct += (torch.max(outputs, dim=1)[1] == labels).sum()
        return total_correct / len(test_set.dataset)

    def save_model(self, model_to_save, epoch, dp=False):
        """Saves a snapshot of a model (as a .pth file) in training.
        This checkpoint can be loaded by calling torch.load(<PATH>)
        ...
        Parameters
        ----------
            model_to_save : torchvision model
                The desired machine learning model to save
            epoch : int
                The current epoch (used for filename)
            dp : bool
                Indicated whether the model was trained using
                DP-SGD or not to add a prefix to the file name
        """

        archictecture_name = str(type(model_to_save)).split(".")[-1].split("'")[0]
        if dp:
            archictecture_name = "DP_" + archictecture_name

        dir_to_save_at = self.prefix + archictecture_name + "_Checkpoints"
        if not os.path.exists(dir_to_save_at):
            os.mkdir(dir_to_save_at)

        file_to_save_at = (
            dir_to_save_at + "/" + archictecture_name + "_" + str(epoch) + ".pth"
        )
        torch.save(model_to_save, file_to_save_at)