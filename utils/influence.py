import numpy as np
import torch
from tqdm import tqdm
from utils import data as data_util
from utils import attacks
import pickle
import os
import torchvision
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
        
def unnormalize(image_tensor, means = (0.4914, 0.4822, 0.4465), stds= (0.2023, 0.1994, 0.2010)):
        
    for i in range(image_tensor.shape[0]):
        mean = means[i]
        std = stds[i]
        image_tensor[i] *= std
        image_tensor[i] += mean
    return torchvision.transforms.ToPILImage()(image_tensor)


def print_augs(augloader, device):
    for x, y in augloader:                
        
        x = x.to(device)
        y = y.to(device)
        
        ims = [unnormalize(x[i]) for i in range(len(x))]
        plt.figure(figsize = (4,4))
        gs1 = gridspec.GridSpec(4, 4)
        gs1.update(wspace=0.025, hspace=0.05) # set the spacing between axes. 

        for i in range(len(x)):
                ax1 = plt.subplot(gs1[i])
                plt.imshow(ims[i])
                plt.axis('on')
                ax1.set_xticklabels([])
                ax1.set_yticklabels([])
                ax1.set_aspect('equal')
                    
#                 plt.savefig("Figures/img20.pdf", bbox_inches='tight')
    
def compute_inNout_sets(challenge_pt, data_logits, mask):
    """Retrives the target model's logits, sets of logits from the IN models and OUT models.

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
           IN models confidences, OUT models confidences wrt ground truth label.
    """
    
    in_mask = mask[challenge_pt].astype(bool)
    out_mask = ~mask[challenge_pt].astype(bool)
    
    in_set = data_logits[:,in_mask]
    out_set = data_logits[:,out_mask]

    return in_set, out_set
    
    
def get_gaus_params(in_dict, out_dict, num_tgt_pts):
    
    num_augs = in_dict[0].shape[0]
    
    in_dist = {i: [None]*num_augs for i in range (num_tgt_pts)}
    out_dist = {i: [None]*num_augs for i in range (num_tgt_pts)}

    for chp_pt in range(num_tgt_pts):
        
        in_mean = torch.mean(in_dict[chp_pt], dim = 1)
        in_std = torch.std(in_dict[chp_pt], dim = 1) + 1e-5
        
        out_mean = torch.mean(out_dict[chp_pt], dim = 1) 
        out_std = torch.std(out_dict[chp_pt], dim = 1) + 1e-5
        
        for aug_idx in range(num_augs): 
            
            in_dist[chp_pt][aug_idx] = torch.distributions.normal.Normal(in_mean[aug_idx], in_std[aug_idx])
            out_dist[chp_pt][aug_idx] = torch.distributions.normal.Normal(out_mean[aug_idx], out_std[aug_idx])   
    
    return in_dist, out_dist


def find_closest_neighbors(refin_logit_dict, refout_logit_dict, in_dist, out_dist, num_tgt_pts, inf_thresh = 0.75):

    nbr_list = {i: [] for i in range(num_tgt_pts)}
    
    num_augs = len(in_dist[0])
    
    #Compute reference distributions
    for chp_pt in range(num_tgt_pts):
        
        in_mean = torch.mean(refin_logit_dict[chp_pt])
        in_std = torch.std(refin_logit_dict[chp_pt]) + 1e-5
        
        out_mean = torch.mean(refout_logit_dict[chp_pt])
        out_std = torch.std(refout_logit_dict[chp_pt]) + 1e-5
        
        ref_in_dist = torch.distributions.normal.Normal(in_mean, in_std)
        ref_out_dist = torch.distributions.normal.Normal(out_mean, out_std)
        
        for aug_idx in range(num_augs):
            
            aug_in_dist = in_dist[chp_pt][aug_idx]
            aug_out_dist = out_dist[chp_pt][aug_idx]
            
            in_kl_score = torch.distributions.kl_divergence(ref_in_dist,aug_in_dist)
            out_kl_score = torch.distributions.kl_divergence(ref_out_dist,aug_out_dist)
            
            #Our metric of sample importance
            if(in_kl_score <= inf_thresh and out_kl_score <= inf_thresh):
                nbr_list[chp_pt].append(aug_idx)
                
    return nbr_list
            
            
class InfluenceMeasure:
    def __init__(
        self,
        experiment_name,
        load_saved=True,
        device="cpu",
        seed=0,
        manual_save_dir=None,
        load_nbrs = False,
    ):
        
        self.name = experiment_name
        if not manual_save_dir:
            self.save_dir = str(type(model)).split(".")[-1].split("'")[0]
        else:
            self.save_dir = manual_save_dir
        self.saved_models_dir = f"ShadowModels/{self.save_dir}_{self.name}"
        
            
        attributes = np.atleast_1d(np.load(f"ShadowModels/{self.save_dir}_{self.name}/{self.name}.npy", allow_pickle=True))[0]
        self.__dict__.update(attributes)
        self.device=device
        self.custom_seed = seed
        self.load_nbrs = load_nbrs
        
        
    def _raw_logits(
        self,
        model, 
        dataloader,
        device,
    ):
        model.eval()
        
        with torch.no_grad():
            model = model.to(device)
            
            # Run in one batch to speed up calculations
            for x, y in dataloader:
                x = x.to(self.device)
                predictions = model(x)
        
        return predictions    
    
    
    def _logit_scaling(
        self,
        model, 
        dataloader,
        device,
    ):
        
        model.eval()
        softmax = torch.nn.Softmax(dim=1)
        with torch.no_grad():
            model = model.to(device)
            
            # Run in one batch to speed up calculations
            for x, y in dataloader:
                
                x = x.to(self.device)
                y = y.to(self.device)
                
                logit_mat = model(x)
                conf_mat = softmax(logit_mat)
                logits = conf_mat[torch.arange(conf_mat.shape[0]),y] #logits wrt to ground truth label
                logits = torch.log(logits+10e-8) - torch.log((1-logits)+10e-8)
                    
            return logits



    def compute_base_logits(self, transform_type):
        """
        Computing logits of the target points.
        """
        
        if(self.load_nbrs):
            return
        
        self.dataset.transform = transform_type
        tmi_dataset = torch.utils.data.Subset(self.dataset, self.target_indices)
        tmi_loader = torch.utils.data.DataLoader(tmi_dataset, batch_size=self.num_target_points, shuffle=False)
        
        # Shadow model observed confidences
        target_point_confs = [torch.Tensor() for _ in range(len(self.target_indices))]
        
        for i in tqdm(
            range(self.num_shadow_models), 
            desc=f"Running Inference on Shadow Models", 
            position=0, 
            leave=True,
        ):
            shadow_model = torch.load(
                f"{self.saved_models_dir}/out_model_{i+1}",
                # f"{self.saved_models_dir}/shadow_model_{i+1}",
                map_location=self.device,
            )
            
            shadow_model.eval()
            sm = torch.nn.Softmax(dim=1)
            raw_outs = self._raw_logits(shadow_model, tmi_loader, self.device).detach().cpu()
            
            outs = sm(raw_outs)
            
            for idx in range(len(outs)):
                target_point_confs[idx] = torch.concatenate([target_point_confs[idx], outs[idx].unsqueeze(0)], dim=0)
                
        target_point_confs = torch.stack(target_point_confs)
        
        self.inbase_logit_dict = {i: torch.Tensor() for i in range (self.num_target_points)}
        self.outbase_logit_dict = {i: torch.Tensor() for i in range (self.num_target_points)}
        
        for challenge_pt in range(self.num_target_points):
            in_set, out_set = attacks.get_in_out_sets(self.dataset, self.target_indices, target_point_confs, challenge_pt, self.mask)
            
            self.inbase_logit_dict[challenge_pt] = in_set
            self.outbase_logit_dict[challenge_pt] = out_set
    
    
    def compute_nbr_logits(self, transform_type, aug_replicas = 1, num_tgt_pts = 0, custom_dataset = None):
        
        assert num_tgt_pts <= self.num_target_points, "The max allowed target points are 500."
        
        if(num_tgt_pts ==0):
            num_tgt_pts = self.num_target_points
        
        if(self.load_nbrs):
            return
        
        # To test for augmentations
        self.data_logits = {}
        
        for chp_pt in tqdm(
                range(num_tgt_pts),
                # range(10,11),
                desc=f"Running Attack on {num_tgt_pts} Challenge Points", 
                position=0, 
                leave=True
            ):
        
            if(aug_replicas >=1):
            
                assert custom_dataset == None, "Dont pass custom dataset while setting aug_replicas to non-zero"

                chp_data = torch.utils.data.Subset(self.dataset, self.target_indices[chp_pt:chp_pt+1])
                tmi_dataset = data_util.AugmentedDataset(mem_data = chp_data, aug_replicas = aug_replicas, 
                                                          transform_type = transform_type)

            elif (aug_replicas == 0):
                assert custom_dataset != None, "aug_replicas =0, pass a custom dataset"
                tmi_dataset = custom_dataset


            dataset_len = len(tmi_dataset)
            data_loader = torch.utils.data.DataLoader(tmi_dataset, batch_size= dataset_len, shuffle = False)

            # Shadow model observed confidences
            sample_logits = [torch.Tensor() for _ in range(dataset_len)]

            for i in range(self.num_shadow_models):
                shadow_model = torch.load(
                    f"{self.saved_models_dir}/out_model_{i+1}",
                    map_location=self.device,
                )

                shadow_model.eval()
                
                outs = self._logit_scaling(shadow_model,data_loader,self.device).detach().cpu()

                for idx in range(len(outs)):
                    sample_logits[idx] = torch.concatenate([sample_logits[idx], outs[idx].unsqueeze(0)], dim=0)

            sample_logits = torch.stack(sample_logits)
            
            self.data_logits[chp_pt] = sample_logits
    
    def get_valid_samples(self, aug_replicas = 1, num_tgt_pts = 0, nbr_thresh = 0.75, save_nbrs = True):
        
        assert num_tgt_pts <= self.num_target_points, "The max allowed target points are 500."
        
        if(num_tgt_pts ==0):
            num_tgt_pts = self.num_target_points
        
       
        nbr_path = f"Neighbors/{self.save_dir}"
        
        nbr_path = nbr_path+"-"+str(nbr_thresh)
        
        if not os.path.exists(nbr_path):
            os.makedirs(nbr_path)
        
        if(aug_replicas >0):
            nbr_path = nbr_path+f"/augs{aug_replicas}" 
        else:
            nbr_path = nbr_path+f"/customdata"
            
        print(nbr_path)
        
        if(self.load_nbrs):
            file = open(nbr_path,"rb")
            nbr_list = pickle.load(file)
            file.close()
            return nbr_list
        
        in_logit_dict = {i: torch.Tensor() for i in range (len(self.data_logits))}
        out_logit_dict = {i: torch.Tensor() for i in range (len(self.data_logits))}
        
        for challenge_pt in tqdm(
            range(num_tgt_pts),
            desc=f"Running Attack on {num_tgt_pts} Target Points", 
            position=0, 
            leave=True
        ):
            
            in_set, out_set = compute_inNout_sets(challenge_pt, self.data_logits[challenge_pt], self.mask)
            
            in_logit_dict[challenge_pt] = in_set
            out_logit_dict[challenge_pt] = out_set
        
        in_dist, out_dist = get_gaus_params(in_logit_dict, out_logit_dict, num_tgt_pts)
        
        nbr_list = find_closest_neighbors(self.inbase_logit_dict, self.outbase_logit_dict, in_dist, out_dist,num_tgt_pts, nbr_thresh)
        
            
        if(save_nbrs):
            assert self.load_nbrs == False, "Set load nbrs to False."
            file = open(nbr_path,"wb")
            pickle.dump(nbr_list, file)
            file.close()
        
        return nbr_list, in_logit_dict, out_logit_dict
            
        
    
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
        