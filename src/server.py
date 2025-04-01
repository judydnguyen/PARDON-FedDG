import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
import wandb

from collections import OrderedDict
import copy

from .finch import FINCH
from .models import *
from .utils import *
from .client import *
from .dataset_bundle import *

import src.adain_net as net

VGG_STATE_DICT_PATH = "src/models/vgg_normalised.pth"
DECODER_STATE_DICT_PATH = "src/models/decoder.pth"

# Load pretrained encoder and decoder from AdaIN

decoder = net.decoder
vgg = net.vgg
decoder.eval()
vgg.eval()

decoder.load_state_dict(torch.load(DECODER_STATE_DICT_PATH))
vgg.load_state_dict(torch.load(VGG_STATE_DICT_PATH))
vgg = nn.Sequential(*list(vgg.children())[:31])

class FedAvg(object):
    def __init__(self, device, ds_bundle, hparam):
        self.ds_bundle = ds_bundle
        self.device = device
        self.clients = []
        self.hparam = hparam
        self.num_rounds = hparam['num_rounds']
        self.fraction = hparam['fraction']
        self.num_clients = 0
        self.test_dataloader = {}
        self._round = 0
        self.featurizer = None
        self.classifier = None
        self.params = hparam
    
    def setup_model(self, model_file=None, start_epoch=0):
        """
        The model setup depends on the datasets. 
        """
        assert self._round == 0
        self._featurizer = self.ds_bundle.featurizer
        self._classifier = self.ds_bundle.classifier
        self.featurizer = nn.DataParallel(self._featurizer)
        self.classifier = nn.DataParallel(self._classifier)
        self.model = nn.DataParallel(nn.Sequential(self._featurizer, self._classifier))
        if model_file:
            self.model.load_state_dict(torch.load(model_file))
            self._round = int(start_epoch)

    def register_clients(self, clients):
        # assert self._round == 0
        self.clients = clients
        self.num_clients = len(self.clients)
        # for client in tqdm(self.clients):
            # client.setup_model(copy.deepcopy(self._featurizer), copy.deepcopy(self._classifier))
    
    def register_testloader(self, dataloaders):
        self.test_dataloader.update(dataloaders)
    
    def transmit_model(self, sampled_client_indices=None):
        """
            Description: Send the updated global model to selected/all clients.
            This method could be overriden by the derived class if one algorithm requires to send things other than model parameters.
        """
        if sampled_client_indices is None:
            # send the global model to all clients before the very first and after the last federated round
            for client in tqdm(self.clients, leave=False):
            # for client in self.clients:
                client.update_model(self.model.state_dict())
        else:
            # send the global model to selected clients
            for idx in tqdm(sampled_client_indices, leave=False):
            # for idx in sampled_client_indices:
                self.clients[idx].update_model(self.model.state_dict())

    def sample_clients(self):
        """
        Description: Sample a subset of clients. 
        Could be overriden if some methods require specific ways of sampling.
        """
        # sample clients randommly
        num_sampled_clients = max(int(self.fraction * self.num_clients), 1)
        sampled_client_indices = sorted(np.random.choice(a=[i for i in range(self.num_clients)], size=num_sampled_clients, replace=False).tolist())

        return sampled_client_indices

    def update_clients(self, sampled_client_indices, flr=0):
        """
        Description: This method will call the client.fit methods. 
        Usually doesn't need to override in the derived class.
        """
        def update_single_client(selected_index):
            self.clients[selected_index].fit(flr=flr)
            client_size = len(self.clients[selected_index])
            return client_size
        selected_total_size = 0
        for idx in tqdm(sampled_client_indices, leave=False):
            client_size = update_single_client(idx)
            selected_total_size += client_size
        return selected_total_size

    def evaluate_clients(self, sampled_client_indices):
        def evaluate_single_client(selected_index):
            self.clients[selected_index].client_evaluate()
            return True
        for idx in tqdm(sampled_client_indices):
            self.clients[idx].client_evaluate()            

    def aggregate(self, sampled_client_indices, coefficients):
        """Average the updated and transmitted parameters from each selected client."""
        averaged_weights = OrderedDict()
        for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
            local_weights = self.clients[idx].model.state_dict()
            for key in self.model.state_dict().keys():
                if it == 0:
                    averaged_weights[key] = coefficients[it] * local_weights[key]
                else:
                    averaged_weights[key] += coefficients[it] * local_weights[key]
        self.model.load_state_dict(averaged_weights)

        for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
            # self.clients[idx].del_model()
            del self.clients[idx].model
            gc.collect()

    def train_federated_model(self, flr=0):
        """Do federated training."""
        # select pre-defined fraction of clients randomly
        sampled_client_indices = self.sample_clients()
        for idx, client in enumerate(self.clients):
            if idx in sampled_client_indices:
                client.setup_model(copy.deepcopy(self._featurizer), copy.deepcopy(self._classifier))
        
        # send global model to the selected clients
        self.transmit_model(sampled_client_indices)

        # updated selected clients with local dataset
        selected_total_size = self.update_clients(sampled_client_indices, flr=flr)

        # evaluate selected clients with local dataset (same as the one used for local update)
        # self.evaluate_clients(sampled_client_indices)

        # average each updated model parameters of the selected clients and update the global model
        mixing_coefficients = [len(self.clients[idx]) / selected_total_size for idx in sampled_client_indices]
        self.aggregate(sampled_client_indices, mixing_coefficients)
    
    def evaluate_global_model(self, dataloader):
        """Evaluate the global model using the global holdout dataset (self.data)."""
        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():
            y_pred = None
            y_true = None
            for batch in tqdm(dataloader):
                data, labels, meta_batch = batch[0], batch[1], batch[2]
                if isinstance(meta_batch, list):
                    meta_batch = meta_batch[0]
                data, labels = data.to(self.device), labels.to(self.device)
                prediction = self.model(data)
                
                if self.ds_bundle.is_classification:
                    prediction = torch.argmax(prediction, dim=-1)
                if y_pred is None:
                    y_pred = prediction
                    y_true = labels
                    metadata = meta_batch
                else:
                    y_pred = torch.cat((y_pred, prediction))
                    y_true = torch.cat((y_true, labels))
                    metadata = torch.cat((metadata, meta_batch))
                # print("DEBUG: server.py:183")
                # break
            metric = self.ds_bundle.dataset.eval(y_pred.to("cpu"), y_true.to("cpu"), metadata.to("cpu"))
            print(metric)
            if self.device == "cuda": torch.cuda.empty_cache()
        self.model.to("cpu")
        return metric[0]

    def fit(self):
        """
        Description: Execute the whole process of the federated learning.
        """
        best_id_val_round = 0
        best_id_val_value = 0
        best_id_val_test_value = 0
        best_lodo_val_round = 0
        best_lodo_val_value = 0
        best_lodo_val_test_value = 0
        saved_model = f'saved_models/best_model_{self.hparam["server_method"]}_{self.hparam["client_method"]}_{self.hparam["dataset"]}.pth'

        for r in range(self.num_rounds):
            print("num of rounds: {}".format(r))
            self._round += 1
            self.train_federated_model(flr=r)
            metric_dict = {}
            id_flag = False
            lodo_flag = False
            id_t_val = 0
            t_val = 0
            lodo_val = 0
            metric_dict["flr"] = r
            id_val = 0
            
            for name, dataloader in self.test_dataloader.items():
                if (name == 'val') and (self.params['val'] == 'False'):
                    continue

                metric = self.evaluate_global_model(dataloader)
                torch.cuda.empty_cache()
                metric_dict[name] = metric
                
                if name == 'val':
                    lodo_val = metric[self.ds_bundle.key_metric]
                if name == 'id_val':
                    id_val = metric[self.ds_bundle.key_metric]
                    if id_val > best_id_val_value:
                        best_id_val_round = r
                        best_id_val_value = id_val
                        id_flag = True
                if name == 'test':
                    t_val = metric[self.ds_bundle.key_metric]
                    print(f"t_val: {t_val}")
                if name == 'id_test':
                    id_t_val = metric[self.ds_bundle.key_metric]
            
            if lodo_flag:
                best_lodo_val_test_value = t_val
            if id_flag:
                best_id_val_test_value = id_t_val
            print(metric_dict)
            wandb.log(metric_dict)
            wandb_logging_item = {
                "flr": r,
                "lodo_val": lodo_val,
                "id_val": id_val,
                "lodo_test": t_val,
                "id_t_val": id_t_val,
            }
            wandb.log({"General_Information/": wandb_logging_item})
            # self.save_model(r)
        if best_id_val_round != 0: 
            wandb.summary['best_id_round'] = best_id_val_round
            wandb.summary['best_id_val_acc'] = best_id_val_test_value
        if best_lodo_val_round != 0:
            wandb.summary['best_lodo_round'] = best_lodo_val_round
            wandb.summary['best_lodo_val_acc'] = best_lodo_val_test_value
        # self.transmit_model()

    def save_model(self, num_epoch):
        path = f"{self.hparam['data_path']}/models/{self.ds_bundle.name}_{self.clients[0].name}_{self.hparam['iid']}_{num_epoch}.pth"
        torch.save(self.model.state_dict(), path)

class FedGMA(FedAvg):
    """
    @article{tenison2022gradient,
    title={Gradient masked averaging for federated learning},
    author={Tenison, Irene and Sreeramadas, Sai Aravind and Mugunthan, Vaikkunth and Oyallon, Edouard and Belilovsky, Eugene and Rish, Irina},
    journal={arXiv preprint arXiv:2201.11986},
    year={2022}
    }
    This code is inherited from original benchmark repo: https://github.com/anonymous-lab-ml/benchmarking-dg-fed
    """
    def register_clients(self, clients):
        self.clients = clients
        self.num_clients = len(self.clients)
        
    def aggregate(self, sampled_client_indices, coefficients):
        """Average the updated and transmitted parameters from each selected client."""
        num_sampled_clients = len(sampled_client_indices)
        delta = []
        sign_delta = ParamDict()
        self.model.to('cpu')
        last_weights = ParamDict(self.model.state_dict())
        for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
            self.clients[idx].model.to('cpu')
            local_weights = ParamDict(self.clients[idx].model.state_dict())
            delta.append(coefficients[it] * (local_weights - last_weights))
            if it == 0:
                sum_delta = delta[it]
                sign_delta = delta[it].sign()
            else:
                sum_delta += delta[it]
                sign_delta += delta[it].sign()
                # if it == 0:
                #     averaged_weights[key] = coefficients[it] * local_weights[key]
                # else:
                #     averaged_weights[key] += coefficients[it] * local_weights[key]
        sign_delta /= num_sampled_clients
        abs_sign_delta = sign_delta.abs()
        # print(sign_delta[key])
        mask = abs_sign_delta.ge(self.hparam['mask_threshold'])
        # print("--mid--")
        # print(mask)
        # print("-------")
        final_mask = mask + (0-mask) * abs_sign_delta
        averaged_weights = last_weights + self.hparam['step_size'] * final_mask * sum_delta 
        self.model.load_state_dict(averaged_weights)
        for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
            del self.clients[idx].model

    def train_federated_model(self, flr=0):
        
        """Do federated training."""
        # select pre-defined fraction of clients randomly
        sampled_client_indices = self.sample_clients()
        for idx, client in enumerate(self.clients):
            if idx in sampled_client_indices:
                client.setup_model(copy.deepcopy(self._featurizer), copy.deepcopy(self._classifier))
        self.transmit_model(sampled_client_indices)
        # updated selected clients with local dataset
        selected_total_size = self.update_clients(sampled_client_indices, flr=flr)

        # evaluate selected clients with local dataset (same as the one used for local update)
        # self.evaluate_clients(sampled_client_indices)

        # average each updated model parameters of the selected clients and update the global model
        mixing_coefficients = [len(self.clients[idx]) / selected_total_size for idx in sampled_client_indices]
        self.aggregate(sampled_client_indices, mixing_coefficients)    
        
class DGServer(FedAvg):
    def __init__(self, device, ds_bundle, validation_dataloader, hparam):
        super().__init__(device, ds_bundle, hparam)
        self.metadata = None
        self.dataset = hparam['dataset']
        self.fixed_meta = self.hparam['fixed_meta']
        self.validation_dataloader = validation_dataloader

    def register_clients(self, clients):
        self.clients = clients
        self.num_clients = len(self.clients)

    def fit(self):
        self.update_metadata() 
        return super().fit()

    def update_metadata(self):
        
        style_path_mean = f'style_stats/{self.dataset}/{self.params["split_scheme"]}_{self.params["iid"]}_mean_arr_{self.num_clients}_clients_FINCH.npy'
        style_path_std = f'style_stats/{self.dataset}/{self.params["split_scheme"]}_{self.params["iid"]}_std_arr_{self.num_clients}_clients_FINCH.npy'
        
        if os.path.exists(style_path_mean):
            
            # np_meta_file: file contains interpolative style statistics
            np_meta_file = f'style_stats/{self.dataset}/{self.params["split_scheme"]}_{self.params["iid"]}_{self.num_clients}_mean_std__FINCH.npy'
            
            if os.path.exists(np_meta_file):
                style_stat = np.load(np_meta_file)
                mean_avg, std_avg = style_stat
                mean_avg, std_avg = torch.Tensor(mean_avg).to(self.device), torch.Tensor(std_avg).to(self.device)
                self.metadata = (mean_avg, std_avg)
                return
            
            mean_style_stat = np.load(style_path_mean)
            std_style_stat = np.load(style_path_std)
            avg_mean, avg_std = self.find_optimal_metadata(mean_style_stat, std_style_stat)
            style_stat = (avg_mean.to(self.device), avg_std.to(self.device))
            self.metadata = style_stat
        
        else:
            meta = []
            for client in self.clients:
                client_metadata = client.abstract_style()
                meta.append(client_metadata)
                
            mean_arr = [i for (i,_) in meta]
            np.save(style_path_mean, torch.cat(mean_arr).cpu().numpy())
            std_arr = [i for (_,i) in meta]
            np.save(style_path_std, torch.cat(std_arr).cpu().numpy())
            
            mean_style_stat = np.load(style_path_mean)
            std_style_stat = np.load(style_path_std)
            avg_mean, avg_std = self.find_optimal_metadata(mean_style_stat, std_style_stat)
            
            self.metadata = (avg_mean.to(self.device), avg_std.to(self.device))
        
    def find_optimal_metadata(self, mean_arr, std_arr):
        
        # mean_arr = torch.tensor(mean_arr)
        # std_arr = torch.tensor(std_arr)
        
        # mean_arr = np.asarray(mean_arr)
        # std_arr = np.asarray(std_arr)
        
        mean_style_stat_reshaped = mean_arr.reshape(mean_arr.shape[0], 512)
        std_style_stat_reshaped = std_arr.reshape(std_arr.shape[0], 512)
        
        # Concatenate mean and std along the last axis
        combined_data = np.hstack([mean_style_stat_reshaped, std_style_stat_reshaped])
        c, num_clust, req_c = FINCH(combined_data, initial_rank=None, req_clust=None, distance='cosine',
                                    ensure_early_exit=False, verbose=True)
        cluster_ids = c[:, 0]
        # Calculate the mean for each cluster
        unique_clusters = np.unique(cluster_ids)
        cluster_means = []

        for cluster_id in unique_clusters:
            cluster_points = combined_data[cluster_ids == cluster_id]
            cluster_mean = np.mean(cluster_points, axis=0)
            cluster_means.append(cluster_mean)

        # Convert cluster_means to a numpy array
        cluster_means = np.array(cluster_means)
        mean_arr = cluster_means[:,:512].reshape(-1, 512, 1, 1)
        std_arr = cluster_means[:,512:].reshape(-1, 512, 1, 1)
        mean_arr = torch.Tensor(mean_arr)
        std_arr = torch.Tensor(std_arr)

        device = torch.device("cuda")
        vgg.to(device)

        avg_mean, _ = torch.median(mean_arr, dim=0, keepdim=True)
        avg_std, _ = torch.median(std_arr, dim=0, keepdim=True)
        
        np.save(f'style_stats/{self.dataset}/{self.params["split_scheme"]}_{self.params["iid"]}_{self.num_clients}_mean_std__FINCH.npy', 
                [avg_mean.cpu().numpy(), avg_std.cpu().numpy()]) 
        return avg_mean, avg_std
    
    def update_clients(self, sampled_client_indices, flr=0):
        """
        Description: This method will call the client.fit methods. 
        Usually doesn't need to override in the derived class.
        """

        def update_single_client(selected_index):
            self.clients[selected_index].contrastive_fit(metadata=self.metadata, flr=flr)
            client_size = len(self.clients[selected_index])
            return client_size
        selected_total_size = 0
        for idx in tqdm(sampled_client_indices, leave=False):
            client_size = update_single_client(idx)
            selected_total_size += client_size
        return selected_total_size    
    
    def train_federated_model(self, flr=0):
        
        """Do federated training."""
        # select pre-defined fraction of clients randomly
        sampled_client_indices = self.sample_clients()
        for idx, client in enumerate(self.clients):
            if idx in sampled_client_indices:
                client.setup_model(copy.deepcopy(self._featurizer), copy.deepcopy(self._classifier))

        self.transmit_model(sampled_client_indices)

        # updated selected clients with local dataset
        selected_total_size = self.update_clients(sampled_client_indices, flr=flr)

        # evaluate selected clients with local dataset (same as the one used for local update)
        # self.evaluate_clients(sampled_client_indices)

        # average each updated model parameters of the selected clients and update the global model
        mixing_coefficients = [len(self.clients[idx]) / selected_total_size for idx in sampled_client_indices]
        self.aggregate(sampled_client_indices, mixing_coefficients)
        
class RethinkFL(FedAvg):
    """ This is the implementation for FPL method
    Cite this: 
    @inproceedings{huang2023rethinking,
    title={Rethinking federated learning with domain shift: A prototype view},
    author={Huang, Wenke and Ye, Mang and Shi, Zekun and Li, He and Du, Bo},
    booktitle={2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    pages={16312--16322},
    year={2023},
    organization={IEEE}
    }
    Github link: https://github.com/WenkeHuang/RethinkFL
    """
    def __init__(self, device, ds_bundle, hparam):
        super().__init__(device, ds_bundle, hparam)
        self.global_protos = []
        self.local_protos = {}
        
    def register_clients(self, clients):
        # assert self._round == 0
        self.clients = clients
        self.num_clients = len(self.clients)
            
    def proto_aggregation(self, sampled_client_indices):
        agg_protos_label = dict()
        for idx in sampled_client_indices:
            local_protos =  self.local_protos[idx]
            for label in local_protos.keys():
                if label in agg_protos_label:
                    agg_protos_label[label].append(local_protos[label])
                else:
                    agg_protos_label[label] = [local_protos[label]]

        for [label, proto_list] in agg_protos_label.items():
            if len(proto_list) > 1:
                proto_list = [item.squeeze(0).detach().cpu().numpy().reshape(-1) for item in proto_list]
                proto_list = np.array(proto_list)

                c, num_clust, req_c = FINCH(proto_list, initial_rank=None, req_clust=None, distance='cosine',
                                            ensure_early_exit=False, verbose=True)

                m, n = c.shape
                class_cluster_list = []
                for index in range(m):
                    class_cluster_list.append(c[index, -1])

                class_cluster_array = np.array(class_cluster_list)
                uniqure_cluster = np.unique(class_cluster_array).tolist()
                agg_selected_proto = []

                for _, cluster_index in enumerate(uniqure_cluster):
                    selected_array = np.where(class_cluster_array == cluster_index)
                    selected_proto_list = proto_list[selected_array]
                    proto = np.mean(selected_proto_list, axis=0, keepdims=True)

                    agg_selected_proto.append(torch.tensor(proto))
                    
                agg_protos_label[label] = agg_selected_proto
            else:
                agg_protos_label[label] = [proto_list[0].data.unsqueeze(0)]
        
        self.global_protos = agg_protos_label
    
    def update_clients(self, sampled_client_indices, flr=0, all_f=None, mean_f=None, all_global_protos_keys=None):
        """
        Description: This method will call the client.fit methods. 
        Usually doesn't need to override in the derived class.
        """
        def update_single_client(selected_index):
            local_protos = self.clients[selected_index].fit(flr=flr, all_global_protos_keys=all_global_protos_keys, all_f=all_f, mean_f=mean_f)
            self.local_protos[selected_index] = local_protos
            client_size = len(self.clients[selected_index])
            return client_size
        
        selected_total_size = 0
        for idx in tqdm(sampled_client_indices, leave=False):
            client_size = update_single_client(idx)
            selected_total_size += client_size
        return selected_total_size
    
    def train_federated_model(self, flr=0):
        """Do federated training."""
        # select pre-defined fraction of clients randomly
        self.local_protos = {}
        sampled_client_indices = self.sample_clients()
        for idx, client in enumerate(self.clients):
            if idx in sampled_client_indices:
                client.setup_model(copy.deepcopy(self._featurizer), copy.deepcopy(self._classifier))
        
        mean_f, all_f, all_global_protos_keys = None, None, None

        if len(self.global_protos) != 0:
            all_global_protos_keys = np.array(list(self.global_protos.keys()))
            all_f = []
            mean_f = []
            for protos_key in all_global_protos_keys:
                temp_f = self.global_protos[protos_key]
                temp_f = torch.cat(temp_f, dim=0).to(self.device)
                all_f.append(temp_f.cpu())
                mean_f.append(torch.mean(temp_f, dim=0).cpu())
            all_f = [item.detach().cpu() for item in all_f]
            mean_f = [item.detach().cpu() for item in mean_f]                                            
        # send global model to the selected clients
        self.transmit_model(sampled_client_indices)
        # updated selected clients with local dataset
        selected_total_size = self.update_clients(sampled_client_indices, flr=flr,
                                                  all_f=all_f, mean_f=mean_f, 
                                                  all_global_protos_keys=all_global_protos_keys)
        
        self.proto_aggregation(sampled_client_indices)
        mixing_coefficients = [len(self.clients[idx]) / selected_total_size for idx in sampled_client_indices]
        self.aggregate(sampled_client_indices, mixing_coefficients)
        
class FedDGGA(FedAvg):
    """
    This is the implementation for FedDGGA method
    Cite this:
    @InProceedings{Zhang_2023_CVPR,
    author    = {Zhang, Ruipeng and Xu, Qinwei and Yao, Jiangchao and Zhang, Ya and Tian, Qi and Wang, Yanfeng},
    title     = {Federated Domain Generalization With Generalization Adjustment},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {3954-3963}
    }
    
    Github: https://github.com/MediaBrain-SJTU/FedDG-GA
    """
    def __init__(self, device, ds_bundle, hparam):
        super().__init__(device, ds_bundle, hparam)
        self.contribution_scores = {}
        self.step_size = 1./3.
        self.total_round = hparam['num_rounds']
        
    def register_clients(self, clients):
        self.clients = clients
        self.num_clients = len(self.clients)
        
    def update_clients(self, sampled_client_indices, flr=0):
        """
        Description: This method will call the client.fit methods. 
        Usually doesn't need to override in the derived class.
        """
        def update_single_client(selected_index):
            local_generalization_gap = self.clients[selected_index].fit(flr=flr)
            client_size = len(self.clients[selected_index])
            return client_size, local_generalization_gap
        
        selected_total_size = 0
        local_generalization_gaps = {}
        for idx in tqdm(sampled_client_indices, leave=False):
            client_size, gap = update_single_client(idx)
            local_generalization_gaps[idx] = gap
            selected_total_size += client_size
        return selected_total_size, local_generalization_gaps
    
    def refine_weight_dict_by_GA(self, sampled_client_indices, local_generalization_gaps, step_size=0.05, fair_metric = "loss", flr=0):
        if fair_metric == 'acc':
            signal = -1.0
        elif fair_metric == 'loss':
            signal = 1.0
        else:
            raise ValueError('fair metric must be acc or loss')
        weight_dict = self.contribution_scores
        value_list = [0.0 for _ in range(len(self.clients))]
        for cli in sampled_client_indices:
            value_list[cli] = local_generalization_gaps[cli]
        value_list = np.array(value_list)
        
        norm_gap_list = value_list/np.max(np.abs(value_list)) # for partitipating clients
        
        for i, cli in enumerate(weight_dict.keys()):
            weight_dict[cli] += signal * norm_gap_list[i] * step_size
        
        self.contribution_scores = self.weight_clip(weight_dict)
        
        coeff_values = [] # this will be used for weighted averaging
        for cli in sampled_client_indices:
            coeff_values.append(self.contribution_scores[cli])

        if sum(coeff_values) == 0:
            coeff_values = [1./len(coeff_values) for _ in coeff_values]
        else:
            coeff_values = [i/sum(coeff_values) for i in coeff_values]
            
        return coeff_values
    
    def weight_clip(self, weight_dict):
        new_total_w = 0.0
        for key_name in weight_dict.keys():
            weight_dict[key_name] = np.clip(weight_dict[key_name], 0.0, 1.0)
            new_total_w += weight_dict[key_name]
        if new_total_w == 0:
            new_total_w = 1.0
        for key_name in weight_dict.keys():
            weight_dict[key_name] /= new_total_w
        return weight_dict
    
    def train_federated_model(self, flr=0):
        # select pre-defined fraction of clients randomly
        sampled_client_indices = self.sample_clients()
        for idx, client in enumerate(self.clients):
            if idx in sampled_client_indices:
                client.setup_model(copy.deepcopy(self._featurizer), copy.deepcopy(self._classifier))
        
        # send global model to the selected clients
        self.transmit_model(sampled_client_indices)
        
        is_converenged = False
        # updated selected clients with local dataset
        selected_total_size, local_generalization_gaps = self.update_clients(sampled_client_indices, flr=flr)
        np_local_generalization_gaps = np.asarray(list(local_generalization_gaps.values()))
        for elem in np_local_generalization_gaps:
            if np.isnan(elem):
                is_converenged = True 
                # to avoiding loss gap is too small, may lead to nan value -> code corruption
                break
        
        cur_step_size = (1-flr/self.total_round)*self.step_size
        
        if flr == 0 or is_converenged:
            mixing_coefficients = [len(self.clients[idx]) / selected_total_size for idx in sampled_client_indices]
        else:
            mixing_coefficients = self.refine_weight_dict_by_GA(sampled_client_indices, local_generalization_gaps, step_size=cur_step_size)
        self.aggregate(sampled_client_indices, mixing_coefficients)                
    
    def fit(self):
        self.contribution_scores = {key: 1/len(self.clients) for key in range(len(self.clients))}
        return super().fit()
class CCST(FedAvg):
    """
    This is the implementation for CCST method
    @citethis
    @inproceedings{chen2023federated,
    title={Federated domain generalization for image recognition via cross-client style transfer},
    author={Chen, Junming and Jiang, Meirui and Dou, Qi and Chen, Qifeng},
    booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
    pages={361--370},
    year={2023}
    }
    """
    def __init__(self, device, ds_bundle, hparam):
        super().__init__(device, ds_bundle, hparam)
        self.style_bank = None
        self.metadata = None
        self.dataset = hparam['dataset']
        self.fixed_meta = self.hparam['fixed_meta']
    
    def register_clients(self, clients):
        # assert self._round == 0
        self.clients = clients
        self.num_clients = len(self.clients)
        self.style_dict = {}
    
    def register_style_bank(self):
        client_style_path = f'style_stats/{self.dataset}/CCST_{self.params["split_scheme"]}_{self.params["iid"]}_mean_arr_{self.num_clients}_clients.npy'
        client_style_path_2 = f'style_stats/{self.dataset}/CCST_{self.params["split_scheme"]}_{self.params["iid"]}_std_arr_{self.num_clients}_clients.npy'
        if os.path.exists(client_style_path):
            mean_style_stat = np.load(client_style_path)
            std_style_stat = np.load(client_style_path_2)
            meta = list(zip(mean_style_stat, std_style_stat))
            self.style_bank = meta
        else:
            meta = []
            for client in self.clients:
                client_metadata = client.abstract_style()
                meta.append(client_metadata)
            mean_arr = [i for (i,_) in meta]
            np.save(client_style_path, torch.cat(mean_arr).cpu().numpy())
            std_arr = [i for (_,i) in meta]
            np.save(client_style_path_2, torch.cat(std_arr).cpu().numpy())
            self.style_bank = meta
    
    def assign_local_styles(self, K=1):
        for cli_idx, client in enumerate(self.clients):
            k_random_style_idxs = random.sample([idx for idx in range(self.num_clients) if idx != cli_idx], K)
            print(f"style idx for client {cli_idx}: {k_random_style_idxs}")
            k_random_styles = [self.style_bank[idx] for idx in k_random_style_idxs]
            client.register_styles(k_random_styles)
            
    def fit(self):
        # print(f"self.fixed_meta: {self.fixed_meta}")
        self.register_style_bank()
        self.assign_local_styles()
        return super().fit()
            
    def update_clients(self, sampled_client_indices, flr=0):
        """
        Description: This method will call the client.fit methods. 
        Usually doesn't need to override in the derived class.
        """

        def update_single_client(selected_index):
            # self.clients[selected_index].fit(flr=flr)
            self.clients[selected_index].transfer_fit(flr=flr)
            client_size = len(self.clients[selected_index])
            return client_size
        selected_total_size = 0
        for idx in tqdm(sampled_client_indices, leave=False):
            client_size = update_single_client(idx)
            selected_total_size += client_size
        return selected_total_size    
    
    def train_federated_model(self, flr=0):
        
        """Do federated training."""
        # select pre-defined fraction of clients randomly
        sampled_client_indices = self.sample_clients()
        for idx, client in enumerate(self.clients):
            if idx in sampled_client_indices:
                client.setup_model(copy.deepcopy(self._featurizer), copy.deepcopy(self._classifier))
        self.transmit_model(sampled_client_indices)

        # updated selected clients with local dataset
        selected_total_size = self.update_clients(sampled_client_indices, flr=flr)

        # average each updated model parameters of the selected clients and update the global model
        mixing_coefficients = [len(self.clients[idx]) / selected_total_size for idx in sampled_client_indices]
        self.aggregate(sampled_client_indices, mixing_coefficients)

    def evaluate_global_model(self, dataloader):
        """Evaluate the global model using the global holdout dataset (self.data)."""
        self.model.eval()
        self.model.to(self.device)
        # cudnn.benchmark = True
        with torch.no_grad():
            y_pred = None
            y_true = None
            for batch in tqdm(dataloader):
                data, labels = batch[0], batch[1]
                data, labels = data.to(self.device), labels.to(self.device)
                prediction = self.model(data)
                
                if self.ds_bundle.is_classification:
                    prediction = torch.argmax(prediction, dim=-1)
                if y_pred is None:
                    y_pred = prediction
                    y_true = labels
                else:
                    y_pred = torch.cat((y_pred, prediction))
                    y_true = torch.cat((y_true, labels))
                # print("DEBUG: server.py:183")
                # break
            metric = self.ds_bundle.dataset.eval(y_pred.to("cpu"), y_true.to("cpu"), torch.Tensor(0))
            print(metric)
            if self.device == "cuda": torch.cuda.empty_cache()
        self.model.to("cpu")
        return metric[0]