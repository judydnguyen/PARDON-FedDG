import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.utils.data import DataLoader, Dataset
from torch.autograd import grad

import torchvision.utils as vutils
import torchvision.transforms as transforms
from wilds.common.data_loaders import get_train_loader

from tqdm.auto import tqdm
from termcolor import colored
import wandb

import gc
import os
import copy
import random

from src.finch import FINCH
from src.utils import *
import src.adain_net as net

VGG_STATE_DICT_PATH = "src/models/vgg_normalised.pth"
DECODER_STATE_DICT_PATH = "src/models/decoder.pth"
decoder = net.decoder
vgg = net.vgg

decoder.eval()
vgg.eval()

decoder.load_state_dict(torch.load(DECODER_STATE_DICT_PATH))
vgg.load_state_dict(torch.load(VGG_STATE_DICT_PATH))
vgg = nn.Sequential(*list(vgg.children())[:31])

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)   

def flatten_jacobian(J):
    return J.view(J.size(0), -1)

def cross_entropy_loss(logits, targets):
    return F.cross_entropy(logits, targets)

def grad_norm_fn(model, X, Y):
    def per_sample_loss_fn(x, y):
        logits = model(x)
        return flatten_jacobian(grad(lambda model_output: cross_entropy_loss(model_output, y), logits, create_graph=True)[0])

    loss_grads = torch.stack([per_sample_loss_fn(x, y) for x, y in zip(X, Y)])
    scores = torch.norm(loss_grads, dim=-1)
    return scores

def lord_error_fn(model, X, Y, ord=2, num_classes=65):
    logits = model(X)
    Y_one_hot = F.one_hot(Y, num_classes=num_classes)  # Assuming model.output_dim is the number of classes
    errors = F.softmax(logits, dim=-1) - Y_one_hot.float()
    scores = torch.norm(errors, p=ord, dim=-1)
    return scores

class CustomDataset(Dataset):
    def __init__(self, data_list, label_list, metadata_list):
        self.data_list = data_list
        self.label_list = label_list
        self.metadata_list = metadata_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        label = self.label_list[idx]
        metadata = []
        return data, label, metadata

class ERM(object):
    """Class for client object having its own (private) data and resources to train a model.

    Participating client has its own dataset which are usually non-IID compared to other clients.
    Each client only communicates with the center server with its trained parameters or globally aggregated parameters.

    Attributes:
        id: Integer indicating client's id.
        data: torch.utils.data.Dataset instance containing local data.
        device: Training machine indicator (e.g. "cpu", "cuda").
        __model: torch.nn instance as a local model.
    """
    def __init__(self, client_id, device, dataset, ds_bundle, hparam):
        """Client object is initiated by the center server."""
        self.client_id = client_id
        self.device = device
        self.featurizer = None
        self.classifier = None
        self.model = None
        self.dataset = dataset # Wrapper of WildSubset.
        self.ds_bundle = ds_bundle
        self.hparam = hparam
        self.n_groups_per_batch = hparam['n_groups_per_batch']
        self.local_epochs = self.hparam['local_epochs']
        self.batch_size = self.hparam["batch_size"]
        self.optimizer_name = self.hparam['optimizer']
        self.optim_config = self.hparam['optimizer_config']
        try:
            self.scheduler_name = self.hparam['scheduler']
            self.scheduler_config = self.hparam['scheduler_config']
        except KeyError:
            self.scheduler_name = 'torch.optim.lr_scheduler.ConstantLR'
            self.scheduler_config = {'factor': 1, 'total_iters': 1}
        self.dataloader = get_train_loader(self.loader_type, self.dataset, batch_size=self.batch_size, uniform_over_groups=None, grouper=self.ds_bundle.grouper, distinct_groups=False, n_groups_per_batch=self.n_groups_per_batch, num_workers=32)
        self.saved_optimizer = False
        self.exp_id = 1

        parent_path = 'local/scratch/a/bai116/opt_dict'
        parent_path_sch = 'local/scratch/a/bai116/sch_dict'

        if not os.path.exists(parent_path):
            os.makedirs(parent_path)
        if not os.path.exists(parent_path_sch):
            os.makedirs(parent_path_sch)
            print("mkdir...")
        self.opt_dict_path = "local/scratch/a/bai116/opt_dict/client_{}.pt".format(self.client_id)
        self.sch_dict_path = "local/scratch/a/bai116/sch_dict/client_{}.pt".format(self.client_id)

    def setup_model(self, featurizer, classifier):
        self._featurizer = featurizer
        self._classifier = classifier
        self.featurizer = nn.DataParallel(self._featurizer)
        self.classifier = nn.DataParallel(self._classifier)
        self.model = nn.DataParallel(nn.Sequential(self._featurizer, self._classifier))

    @property
    def loader_type(self):
        return 'standard'

    def update_model(self, model_dict):
        self.model.load_state_dict(model_dict)
    
    def init_train(self):
        self.model.train()
        self.model.to(self.device)
        self.optimizer = eval(self.optimizer_name)(self.model.parameters(), **self.optim_config)
        self.scheduler = eval(self.scheduler_name)(self.optimizer, **self.scheduler_config)
        if self.saved_optimizer:
            self.optimizer.load_state_dict(torch.load(self.opt_dict_path))
            self.scheduler.load_state_dict(torch.load(self.sch_dict_path))
    
    def end_train(self):
        self.optimizer.zero_grad(set_to_none=True)
        self.model.to("cpu")
        # torch.save(self.optimizer.state_dict(), self.opt_dict_path)
        # torch.save(self.scheduler.state_dict(), self.sch_dict_path)
        del self.scheduler, self.optimizer

    def fit(self, flr=0):
        """Update local model using local dataset."""
        self.init_train()
        for e in range(self.local_epochs):
            for batch in tqdm(self.dataloader):
                results = self.process_batch(batch)
                self.step(results, flr=flr)
            # print("DEBUG: client.py:101")
            # break
        self.end_train()
        self.model.to('cpu')

    def evaluate(self):
        """Evaluate local model using local dataset (same as training set for convenience)."""
        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():
            metric = {}
            y_pred = None
            y_true = None
            for batch in self.dataloader:
                results = self.process_batch(batch)
                if y_pred is None:
                    y_pred = results['y_pred']
                    y_true = results['y_true']
                else:
                    y_pred = torch.cat((y_pred, results['y_pred']))
                    y_true = torch.cat((y_true, results['y_true']))
            metric_new = self.dataset.eval(torch.argmax(y_pred, dim=-1).to("cpu"), y_true.to("cpu"), results["metadata"].to("cpu"))
            for key, value in metric_new[0].item():
                if key not in metric.keys():
                    metric[key] = value
                else:
                    metric[key] += value
        
                if self.device == "cuda": torch.cuda.empty_cache()
        self.model.to('cpu')
        return metric
    
    def calc_loss(self):
        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():
            y_pred = None
            y_true = None
            for batch in self.dataloader:
                results = self.process_batch(batch)
                if y_pred is None:
                    y_pred = results['y_pred']
                    y_true = results['y_true']
                else:
                    y_pred = torch.cat((y_pred, results['y_pred']))
                    y_true = torch.cat((y_true, results['y_true']))
            loss = self.ds_bundle.loss.compute(y_pred, y_true, return_dict=False).mean()

        self.model.to('cpu')
        return loss
    
    def process_batch(self, batch):
        x, y_true, metadata = batch
        x = x.to(self.device)
        y_true = y_true.to(self.device)
        g = self.ds_bundle.grouper.metadata_to_group(metadata).to(self.device)
        metadata = metadata.to(self.device)
        outputs = self.model(x)
        # print(outputs.shape)
        results = {
            'g': g,
            'y_true': y_true,
            'y_pred': outputs,
            'metadata': metadata,
        }
        return results
    
    def process_batch_synz(self, batch):
        x, y_true, metadata = batch
        x = x.to(self.device)
        y_true = y_true.to(self.device)
        # g = self.ds_bundle.grouper.metadata_to_group(metadata).to(self.device)
        # metadata = metadata.to(self.device)
        outputs = self.model(x)
        # print(outputs.shape)
        results = {
            # 'g': g,
            'y_true': y_true,
            'y_pred': outputs,
            'metadata': metadata,
        }
        return results

    def step(self, results, flr=0):
        # print(results['y_true'])
        # objective = eval(self.criterion)()(results['y_pred'], results['y_true'])
        objective = self.ds_bundle.loss.compute(results['y_pred'], results['y_true'], return_dict=False).mean()
        wandb.log({"loss/{}".format(self.client_id): objective.item(), "flr": flr})
        if objective.grad_fn is None:
            pass
        try:
            objective.backward()
        except RuntimeError:
            print(objective)
            print(objective.grad_fn)
        self.optimizer.step()
        self.optimizer.zero_grad()

    @property
    def name(self):
        return self.__class__.__name__
    
    def __len__(self):
        """Return a total size of the client's local data."""
        return len(self.dataset)

class FedSR(ERM):
    """
    @article{nguyen2022fedsr,
    title={Fedsr: A simple and effective domain generalization method for federated learning},
    author={Nguyen, A Tuan and Torr, Philip and Lim, Ser Nam},
    journal={Advances in Neural Information Processing Systems},
    volume={35},
    pages={38831--38843},
    year={2022}
    }
    """
    def __init__(self, client_id, device, dataset, ds_bundle, hparam): 
        super().__init__(client_id, device, dataset, ds_bundle, hparam)
        self.l2_regularizer = hparam['hparam1']
        self.cmi_regularizer = hparam['hparam2']
        self.fp = "local/scratch/a/bai116/tmp/fedsr_ref_exp{}_client_{}.pt".format(self.exp_id, self.client_id)
    
    def setup_model(self, featurizer, classifier):
        super().setup_model(featurizer, classifier)
        self.reference_params = nn.Parameter(torch.ones(self.ds_bundle.n_classes, 2*self._featurizer.n_outputs, device=self.device))
        torch.save(self.reference_params, self.fp)
        del self.reference_params
    
    def del_model(self):
        del self.model
        gc.collect()

    def init_train(self):
        self.reference_params = torch.load(self.fp)
        self.model.train()
        self.model.to(self.device)
        self.optimizer = eval(self.optimizer_name)(list(self.model.parameters())+[self.reference_params], **self.optim_config)
        # if self.saved_optimizer:
        #     self.optimizer.load_state_dict(torch.load(self.opt_dict_path))
            
    def end_train(self):
        self.optimizer.zero_grad(set_to_none=True)
        self.model.to("cpu")
        # torch.save(self.optimizer.state_dict(), self.opt_dict_path)
        torch.save(self.reference_params, self.fp)
        del self.reference_params, self.optimizer

    @property
    def loader_type(self):
        return 'standard'

    def process_batch(self, batch):
        """
        Overrides single_model_algorithm.process_batch().
        Args:
            - batch (tuple of Tensors): a batch of data yielded by data loaders
            - unlabeled_batch (tuple of Tensors or None): a batch of data yielded by unlabeled data loader
        Output:
            - results (dictionary): information about the batch
                - y_true (Tensor): ground truth labels for batch
                - g (Tensor): groups for batch
                - metadata (Tensor): metadata for batch
                - unlabeled_g (Tensor): groups for unlabeled batch
                - features (Tensor): featurizer output for batch and unlabeled batch
                - y_pred (Tensor): full model output for batch and unlabeled batch
        """
        # forward pass
        x, y_true, metadata = batch
        x = x.to(self.device)
        y_true = y_true.to(self.device)
        g = self.ds_bundle.grouper.metadata_to_group(metadata).to(self.device)
        metadata = metadata.to(self.device)
        results = {
            'g': g,
            'y_true': y_true,
            'metadata': metadata,
        }
        features_params = self.featurizer(x)
        z_dim = int(features_params.shape[-1]/2)
        if len(features_params.shape) == 2:
            z_mu = features_params[:,:z_dim]
            z_sigma = F.softplus(features_params[:,z_dim:])
            z_dist = dist.Independent(dist.normal.Normal(z_mu,z_sigma),1)
            features = z_dist.rsample()
        elif len(features_params.shape) == 3:
            flattened_features_params = features_params.view(-1, features_params.shape[-1])
            z_mu = flattened_features_params[:,:z_dim]
            z_sigma = F.softplus(flattened_features_params[:,z_dim:])
            z_dist = dist.Independent(dist.normal.Normal(z_mu,z_sigma),1)
            features = z_dist.rsample()
            features = features.view(x.shape[0], -1, z_dim)
        y_pred = self.classifier(features)
        results['features'] = features
        results['z_mu'] = z_mu
        results['z_sigma'] = z_sigma
        results['feature_params'] = features_params
        results['y_pred'] = y_pred
        return results

    def l2_penalty(self, features):
        if self.ds_bundle.name == 'py150':
            num_samples = features.shape[0] * features.shape[1]
        else:
            num_samples = features.shape[0]
        return torch.sum(features ** 2) / num_samples
    
    def cmi_penalty(self, y, z_mu, z_sigma):
        num_samples = y.shape[0]
        dimension = self.reference_params.shape[1] // 2
        if self.ds_bundle.name == 'py150':
            is_labeled = ~torch.isnan(y)
            flattened_y = y[is_labeled]
            z_mu = z_mu[is_labeled.view(-1)]
            z_sigma = z_sigma[is_labeled.view(-1)]
            target_mu = self.reference_params[flattened_y.to(dtype=torch.long), :dimension]
            target_sigma = F.softplus(self.reference_params[flattened_y.to(dtype=torch.long), dimension:])
        else:
            target_mu = self.reference_params[y.to(dtype=torch.long), :dimension]
            target_sigma = F.softplus(self.reference_params[y.to(dtype=torch.long), dimension:])
        cmi_loss = torch.sum((torch.log(target_sigma) - torch.log(z_sigma) + (z_sigma ** 2 + (target_mu - z_mu) ** 2) / (2*target_sigma**2) - 0.5)) / num_samples
        return cmi_loss

    def step(self, results, flr=0):
        loss = self.ds_bundle.loss.compute(results['y_pred'], results['y_true'], return_dict=False).mean()  
        l2_loss = self.l2_penalty(results["features"])
        cmi_loss = self.cmi_penalty(results["y_true"], results["z_mu"], results["z_sigma"])
        wandb.log({"loss/{}".format(self.client_id): loss.item(), "flr": flr,
                   "l2_loss/{}".format(self.client_id): l2_loss.item(), 
                   "cmi_loss/{}".format(self.client_id): cmi_loss.item()})
        # print(loss.item(), l2_loss.item(), cmi_loss.item())
        self.optimizer.zero_grad()
        (loss + self.l2_regularizer * l2_loss + self.cmi_regularizer * cmi_loss).backward()
        # loss.backward()
        self.optimizer.step()
        del results
        gc.collect()

class DGClient(ERM):
    def __init__(self, client_id, device, dataset, ds_bundle, hparam):
        super().__init__(client_id, device, dataset, ds_bundle, hparam)
        self.mu = self.hparam['hparam1']
        self.gamma = self.hparam['param_1'] # 0.5, param for style-transfer AdaIN
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.output_size = self.hparam['output_size']
        self.infoNCET = self.hparam['infoNCET']
        self.total_round = hparam['num_rounds']
        self.beta = 1.0
        self.custom_dataloader = None
    
    def init_train(self):
        self.model.train()
        self.model.to(self.device)
        self.global_model = copy.deepcopy(self.model)
        self.optimizer = eval(self.optimizer_name)(self.model.parameters(), **self.optim_config)
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler = eval(self.scheduler_name)(self.optimizer, **self.scheduler_config)
        
        if self.saved_optimizer:
            self.optimizer.load_state_dict(torch.load(self.opt_dict_path))
            self.scheduler.load_state_dict(torch.load(self.sch_dict_path))
        
    def end_train(self):
        self.optimizer.zero_grad(set_to_none=True)
        self.model.to("cpu")
        torch.save(self.optimizer.state_dict(), self.opt_dict_path)
        torch.save(self.scheduler.state_dict(), self.sch_dict_path)
        del self.scheduler, self.optimizer, self.global_model
    
    def del_model(self):
        del self.model
    
    def step(self, objective, flr=0):
        wandb.log({"loss/{}".format(self.client_id): objective.item(),
                   "flr": flr})
        if objective.grad_fn is None:
            pass
        try:
            objective.backward()
        except RuntimeError:
            print(objective)
            print(objective.grad_fn)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def contrastive_fit(self, output_size=None, metadata=None, flr=0):
        """Update local model using local dataset."""
        self.beta = (1-flr/self.total_round)*self.beta
        self.init_train()
        
        # Load the pretrained encoder
        vgg.to(self.device)
        feat_mean, feat_std = metadata
        style_stat = [feat_mean, feat_std]
        decoder.to(self.device)

        for e in range(self.local_epochs):
            batch_id = 0
            for batch in tqdm(self.dataloader):
                self.optimizer.zero_grad(set_to_none=True)
                images, labels, metadata = batch
                transfered_imgs = self.style_stransfer_batch(batch, alpha=0.5, style_stat=style_stat)
                transfered_imgs = transfered_imgs.to(self.device)
                transfered_imgs.requires_grad_()
                images = images.to(self.device)
                labels = labels.to(self.device)
                f = self.featurizer(images)
                f_p = self.featurizer(transfered_imgs)
                reg = torch.mean(f**2+f_p**2)
                outputs = self.classifier(f)
                loss_CE = self.ds_bundle.loss.compute(outputs, labels, return_dict=False).mean()
                loss_CON = 0.0
                
                if self.gamma != 0.0:
                    loss_CON = self.contrastive_loss(batch, transfered_imgs)
                
                loss = loss_CE + 0.75*loss_CON + 0.2*reg # NOTE: 0.75 for loss_CON, reg = 0.2 for PACS, 0.1 for OfficeHome
                self.step(loss, flr)
                batch_id += 1
                del f, f_p
                gc.collect()
        self.end_train()

    def style_stransfer_batch(self, batch, alpha=0.5, output_size=224, style_stat=None):
        # This function conducts style-transferring for each batch
        transform = transforms.Grayscale(num_output_channels=1)
        resize = transforms.Resize(self.output_size)
        cp_batch = copy.deepcopy(batch)
        # The above code is not doing anything. It only contains the word "data" and three hash
        # symbols, which are used to create comments in Python code.
        data, y_true, metadata = cp_batch
        original_img = data.clone().cpu()
        
        if self.hparam['dataset'] == "FEMNIST":
            data = data.expand(-1, 3, -1, -1)
            alpha = 0.2
        style_stat = [stat for stat in style_stat]

        with torch.no_grad():
            output = style_transfer(vgg=vgg, decoder=decoder, content=data, style_stat=style_stat, alpha=alpha, device=self.device)
        if output_size > 0:
            output = resize(output)
            
        # Convert it back to a single-channel image if needed
        if self.hparam['dataset'] == "FEMNIST":
            img_single_channel = transform(output)
        else:
            img_single_channel = copy.deepcopy(output)
            
        batch_img = torch.cat(
        [original_img[:8].clone().cpu(), img_single_channel[:8].clone().cpu()], 0)
        grid = vutils.make_grid(batch_img, nrow=8, padding=2, normalize=True)
        
        parent_dir = f"src/track_imgs/{self.hparam['dataset']}/{self.hparam['split_scheme']}"
        # save style-transferred images, just for visualization
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        output_path = f"{parent_dir}/FedTrans__{self.client_id}_output_grid.png"
        vutils.save_image(grid, output_path)
        output = img_single_channel.cpu()
        return output
        
    def contrastive_loss(self, batch, transfered_imgs):
        
        triplets = self.sample_local_triplets(batch, transfered_imgs)
        if not len(triplets):
            return 0.0
        anchor, positive, negative = zip(*triplets)
        triplet_loss = nn.TripletMarginLoss(margin=0.3, p=2, eps=1e-7)
        
        anchor = self.featurizer(torch.stack(anchor))
        positive = self.featurizer(torch.stack(positive))
        negative = self.featurizer(torch.stack(negative))
        
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        negative = F.normalize(negative, p=2, dim=1)
        
        loss2 = triplet_loss(anchor, positive, negative)
        return loss2
    
    def process_batch(self, batch):
        """
        Overrides single_model_algorithm.process_batch().
        Args:
            - batch (tuple of Tensors): a batch of data yielded by data loaders
            - unlabeled_batch (tuple of Tensors or None): a batch of data yielded by unlabeled data loader
        Output:
            - results (dictionary): information about the batch
                - y_true (Tensor): ground truth labels for batch
                - g (Tensor): groups for batch
                - metadata (Tensor): metadata for batch
                - unlabeled_g (Tensor): groups for unlabeled batch
                - features (Tensor): featurizer output for batch and unlabeled batch
                - y_pred (Tensor): full model output for batch and unlabeled batch
        """
        # forward pass
        x, y_true, metadata = batch
        y_true = y_true.to(self.device)
        g = self.ds_bundle.grouper.metadata_to_group(metadata).to(self.device)
        metadata = metadata.to(self.device)
        results = {
            'g': g,
            'y_true': y_true,
            'metadata': metadata,
        }
        x = x.to(self.device)
        features = self.featurizer(x)
        outputs = self.classifier(features)
        y_pred = outputs[: len(y_true)]
        results['features'] = features
        results['y_pred'] = y_pred
        return results
    
    def setup_model(self, featurizer, classifier):
        self._featurizer = featurizer
        self._classifier = classifier
        self.featurizer = nn.DataParallel(self._featurizer)
        self.classifier = nn.DataParallel(self._classifier)
        self.model = nn.DataParallel(nn.Sequential(self._featurizer, self._classifier))

    def sample_local_triplets(self, batch, transfered_img):
        anchors, labels, _ = batch
        triplets = []

        styled_anchors = transfered_img.to(self.device)
        anchors = anchors.to(self.device)
        for i, anchor in enumerate(anchors):
            label = labels[i]

            # Get positive example
            positive_idx = i
            positive = styled_anchors[positive_idx]
            
            negative_indices = [j for j, l in enumerate(labels) if l != label]
            if len(negative_indices) == 0:
                continue  # Skip if there are no negative examples with a different label
            
            closest_negative = styled_anchors[random.choice(negative_indices)]
            # Create triplets
            triplets.append((anchor, positive, closest_negative))
        return triplets
    
    def abstract_style(self):
        # This func extract local style info
        all_feat_sum, all_feat_square_sum, all_count, img_count = 0, 0, 0, 0
        device = torch.device("cuda")
        vgg.to(device)
        modules = list(vgg.children())[:-1]
        feature_extractor = nn.Sequential(*modules)
        feature_extractor.eval()
        print(colored(f"Abstracting information of client {self.client_id}", "blue"))
        all_feat = None  # Initialize to None
        all_feat_list = []
        with torch.no_grad():
            for it, batch in enumerate(self.dataloader):
                data, y_true, metadata = batch
                data = data.to(device)
                data = data.expand(-1, 3, -1, -1)
                feat = feature_extractor(data)  # torch.Size([4, 512, 64, 64])
                img_count += len(data)
                all_feat_list.append(feat.detach())
                del data
        
        all_feat = torch.cat(all_feat_list, dim=0)
        all_feat_rs = all_feat.cpu().data.numpy().reshape(all_feat.shape[0], -1)
        c, num_clust, req_c = FINCH(all_feat_rs, initial_rank=None, req_clust=None, distance='cosine',
                                            ensure_early_exit=False, verbose=True)
        m, n = c.shape
        style_cluster_list = []
        
        for index in range(m):
            style_cluster_list.append(c[index, -1])
        unique_cluster = np.unique(style_cluster_list).tolist()
        
        style_cluster_list = np.asarray(style_cluster_list)
        for _, cluster_idx in enumerate(unique_cluster):
            selected_array = (np.where(style_cluster_list == cluster_idx))[0].tolist()
            selected_feat_list = all_feat[selected_array]
            feat_sum, feat_square_sum, count = calc_sum(selected_feat_list)
            feat_sum, feat_square_sum = feat_sum / float(count), feat_square_sum / float(count)
            all_feat_sum += feat_sum
            all_feat_square_sum += feat_square_sum
            all_count += 1
            del selected_feat_list
            
        feat_mean = all_feat_sum / float(all_count)
        feat_var = all_feat_square_sum / float(all_count) - feat_mean ** 2
        feat_std = torch.sqrt(feat_var + 1e-5)
        
        del all_feat_list
        del all_feat_rs
        gc.collect()
        return feat_mean, feat_std

class FPLClient(ERM):
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
    def __init__(self, client_id, device, dataset, ds_bundle, hparam):
        super().__init__(client_id, device, dataset, ds_bundle, hparam)
        self.agg_protos = None
        self.infoNCET = hparam['infoNCET']
        
    def hierarchical_info_loss(self, f_now, label, all_f, mean_f, all_global_protos_keys):
        pos_idx = np.where(all_global_protos_keys == label.item())[0]
        if pos_idx:
            f_pos = all_f[pos_idx[0]].to(self.device)
        else:
            f_pos = None
        neg_idx = np.where(all_global_protos_keys != label.item())[0]
        f_neg = []
        for idx in neg_idx:
            if all_f[idx].dim() == 1:
                f_neg.append(all_f[idx].to(self.device))
            elif all_f[idx].dim() == 2:
                f_neg.append(all_f[idx][0].to(self.device))

        for idx, elem in enumerate(f_neg):
            if elem.dim() == 1:
                f_neg[idx] = elem.unsqueeze(0)
        f_neg = torch.cat(f_neg)
        xi_info_loss = self.calculate_infonce(f_now, f_pos, f_neg)
        mean_f_pos = mean_f[pos_idx[0]].to(self.device)
        mean_f_pos = mean_f_pos.view(1, -1)

        loss_mse = nn.MSELoss()
        cu_info_loss = loss_mse(f_now, mean_f_pos)
        
        hierar_info_loss = cu_info_loss
        if xi_info_loss is not None:
            hierar_info_loss = xi_info_loss + cu_info_loss
        return hierar_info_loss
    
    def calculate_infonce(self, f_now, f_pos, f_neg):
        if f_pos is None:
            return None
        infonce_loss = 0
        f_proto = torch.cat((f_pos, f_neg), dim=0)
        l = torch.cosine_similarity(f_now, f_proto, dim=1)
        l = l / self.infoNCET

        exp_l = torch.exp(l)
        exp_l = exp_l.view(1, -1)
        pos_mask = [1 for _ in range(f_pos.shape[0])] + [0 for _ in range(f_neg.shape[0])]
        pos_mask = torch.tensor(pos_mask, dtype=torch.float).to(self.device)
        pos_mask = pos_mask.view(1, -1)
        pos_l = exp_l * pos_mask
        sum_pos_l = pos_l.sum(1)
        sum_exp_l = exp_l.sum(1)
        
        if sum_exp_l:
            infonce_loss = -torch.log(sum_pos_l / sum_exp_l)
        return infonce_loss
    
    def setup_model(self, featurizer, classifier):
        self._featurizer = featurizer
        self._classifier = classifier
        self.featurizer = nn.DataParallel(self._featurizer)
        self.classifier = nn.DataParallel(self._classifier)
        self.model = nn.DataParallel(nn.Sequential(self._featurizer, self._classifier))
    
    def fit(self, flr=0, all_global_protos_keys=[], all_f=None, mean_f=None):
        self.init_train()
        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)
        
        iterator = tqdm(range(self.local_epochs))
        for iter in iterator:
            agg_protos_label = {}
            for _, (batch) in enumerate(self.dataloader):
                self.optimizer.zero_grad(set_to_none=True)
                images, labels, _ = batch
                images = images.to(self.device)
                labels = labels.to(self.device)
                f = self.featurizer(images)

                outputs = self.classifier(f)
                lossCE = criterion(outputs, labels)
                if all_global_protos_keys is None:
                    loss_InfoNCE = 0 * lossCE
                elif len(all_global_protos_keys) > 0:
                    i = 0
                    loss_InfoNCE = 0.0
                    loss_InfoNCEs = []
                    
                    for label in labels:
                        if label.item() in all_global_protos_keys:
                            f_now = f[i].unsqueeze(0)
                            loss_instance = self.hierarchical_info_loss(f_now, label, all_f, mean_f, all_global_protos_keys)
                            loss_InfoNCEs.append(loss_instance)
                        i+= 1
                    if len(loss_InfoNCEs):     
                        loss_InfoNCE = sum(loss_InfoNCEs)/len(loss_InfoNCEs)
                loss_InfoNCE = loss_InfoNCE
                loss = lossCE + loss_InfoNCE
                loss.backward()
                log_loss_InfoNCE = 0 if not loss_InfoNCE else loss_InfoNCE.item()
                
                wandb.log({"loss/{}".format(self.client_id): loss.item(), "flr": flr,
                   "lossCE/{}".format(self.client_id): lossCE.item(), 
                   "loss_InfoNCE/{}".format(self.client_id): log_loss_InfoNCE})
                self.optimizer.step()
                
                if iter == self.local_epochs - 1:
                    for i in range(len(labels)):
                        if labels[i].item() in agg_protos_label:
                            agg_protos_label[labels[i].item()].append(f[i, :])
                        else:
                            agg_protos_label[labels[i].item()] = [f[i, :]]
                # del f
                gc.collect()
        print("Local Pariticipant %d CE = %0.3f,InfoNCE = %0.3f" % (self.client_id, lossCE, loss_InfoNCE))
        agg_protos = self.agg_func(agg_protos_label)
        self.end_train()
        self.model.to('cpu')
        return agg_protos
    
    def agg_func(self, protos):
        """
        Returns the average of the weights.
        """
        for [label, proto_list] in protos.items():
            if len(proto_list) > 1:
                proto = 0 * proto_list[0].data
                for i in proto_list:
                    proto += i.data
                protos[label] = proto / len(proto_list)
            else:
                protos[label] = proto_list[0]
        return protos

class FedDGGAClient(ERM):
    
    def __init__(self, client_id, device, dataset, ds_bundle, hparam):
        super().__init__(client_id, device, dataset, ds_bundle, hparam)
        self.historical_empirical_losses = []
        self.criterion = nn.CrossEntropyLoss().to(self.device)
    
    def site_evaluation(self, criterion=None):
        self.model.eval()
        total_loss = 0.0
        total_count = 0.0
        correct_count = 0.0
        pred_list = []
        label_list = []
        with torch.no_grad():
            for batch in tqdm(self.dataloader):
                x, y_true, metadata = batch
                x = x.to(self.device)
                output = self.model(x).cpu()
                loss = criterion(output, y_true)*len(y_true)
                total_loss += loss
                pred = output.data.max(1)[1]
                pred_list.extend(pred.cpu().numpy())
                label_list.extend(y_true.cpu().numpy())
                correct_count += pred.eq(y_true.data.view_as(pred)).sum()
                total_count += len(y_true)
        result_dict = {}
        result_dict['acc'] = float(correct_count)/float(total_count)
        result_dict['loss'] = float(total_loss)/float(total_count)
        return result_dict
    
    def step(self, results, flr=0):
        objective = self.ds_bundle.loss.compute(results['y_pred'], results['y_true'], return_dict=False).mean()
        total_cnt = len(results['y_pred'])
        wandb.log({"loss/{}".format(self.client_id): objective.item(), "flr": flr})
        if objective.grad_fn is None:
            pass
        try:
            objective.backward()
        except RuntimeError:
            print(objective)
            print(objective.grad_fn)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return objective.item()*total_cnt, total_cnt
                
    def fit(self, flr=0):
        """Update local model using local dataset."""
        self.init_train()
        last_local_result_before_lc_train = {}
        global_result_before_lc_train = self.site_evaluation(self.criterion)
        if len(self.historical_empirical_losses):
            last_local_result_before_lc_train = self.historical_empirical_losses[-1]
        else:
            last_local_result_before_lc_train['loss'] = 0.0
        generalization_gap = global_result_before_lc_train['loss'] - last_local_result_before_lc_train['loss']
        total_loss = 0.0
        total_cnt = 0.0
        for e in range(self.local_epochs):
            for batch in tqdm(self.dataloader):
                self.optimizer.zero_grad(set_to_none=True)
                results = self.process_batch(batch)
                loss, cnt = self.step(results, flr=flr)
                total_loss += loss
                total_cnt += cnt
        client_loss = total_loss/total_cnt
        self.end_train()
        result_after_lc_train = {}
        result_after_lc_train['loss'] = client_loss
        self.historical_empirical_losses.append(result_after_lc_train)
        return generalization_gap
    
class CCSTClient(ERM):
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
    def __init__(self, client_id, device, dataset, ds_bundle, hparam):
        super().__init__(client_id, device, dataset, ds_bundle, hparam)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.output_size = self.hparam['output_size']
        self.infoNCET = self.hparam['infoNCET']
        self.total_round = hparam['num_rounds']
        self.style_bank = []
        self.alpha = 0.75
        if self.output_size > 0:
            self.resize = transforms.Resize(self.output_size)
    
    def init_train(self):
        self.model.train()
        self.model.to(self.device)
        self.global_model = copy.deepcopy(self.model)
        self.optimizer = eval(self.optimizer_name)(self.model.parameters(), **self.optim_config)
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler = eval(self.scheduler_name)(self.optimizer, **self.scheduler_config)
        
        if self.saved_optimizer:
            self.optimizer.load_state_dict(torch.load(self.opt_dict_path))
            self.scheduler.load_state_dict(torch.load(self.sch_dict_path))
        
    def end_train(self):
        self.optimizer.zero_grad(set_to_none=True)
        self.model.to("cpu")
        torch.save(self.optimizer.state_dict(), self.opt_dict_path)
        torch.save(self.scheduler.state_dict(), self.sch_dict_path)
        del self.scheduler, self.optimizer, self.global_model

    def style_stransfer_batch(self, batch, alpha=0.5, output_size=224, style_stat=None):
        transform = transforms.Grayscale(num_output_channels=1)
        resize = transforms.Resize(self.output_size)
        cp_batch = copy.deepcopy(batch)
        # The above code is not doing anything. It only contains the word "data" and three hash
        # symbols, which are used to create comments in Python code.
        data, y_true, metadata = cp_batch
        original_img = data.clone().cpu()
        
        if self.hparam['dataset'] == "FEMNIST":
            data = data.expand(-1, 3, -1, -1)
            alpha = 0.2
        style_stat = [stat for stat in style_stat]

        with torch.no_grad():
            output = style_transfer(vgg=vgg, decoder=decoder, content=data, style_stat=style_stat, alpha=alpha, device=self.device)
        if output_size > 0:
            output = resize(output)
        # Convert it back to a single-channel image
        # img_single_channel = torch.mean(output, dim=1, keepdim=True)
        if self.hparam['dataset'] == "FEMNIST":
            img_single_channel = transform(output)
        else:
            img_single_channel = copy.deepcopy(output)

        batch_img = torch.cat(
        [original_img[:8].clone().cpu(), img_single_channel[:8].clone().cpu()], 0)
        grid = vutils.make_grid(batch_img, nrow=8, padding=2, normalize=True)
        
        parent_dir = f"src/tracks/{self.hparam['dataset']}/{self.hparam['split_scheme']}"
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        output_path = f"{parent_dir}/CCST_info__{self.client_id}_output_grid.pdf"
        vutils.save_image(grid, output_path)
        output = img_single_channel.cpu()
        return output
    
    def register_styles(self, k_random_styles):
        self.style_bank = k_random_styles
        
    def del_model(self):
        del self.model
    
    def step(self, results, flr=0):
        # print(results['y_true'])
        # objective = eval(self.criterion)()(results['y_pred'], results['y_true'])
        objective = self.ds_bundle.loss.compute(results['y_pred'], results['y_true'], return_dict=False).mean()
        wandb.log({"loss/{}".format(self.client_id): objective.item(), "flr": flr})
        if objective.grad_fn is None:
            pass
        try:
            objective.backward()
        except RuntimeError:
            print(objective)
            print(objective.grad_fn)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def step_loss(self, objective, flr=0):
        # print(results['y_true'])
        # objective = eval(self.criterion)()(results['y_pred'], results['y_true'])
        # objective = self.ds_bundle.loss.compute(results['y_pred'], results['y_true'], return_dict=False).mean() + self.mu / 2 * self.prox()
        wandb.log({"loss/{}".format(self.client_id): objective.item(),
                   "flr": flr})
        if objective.grad_fn is None:
            pass
        try:
            objective.backward()
        except RuntimeError:
            print(objective)
            print(objective.grad_fn)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def process_batch(self, batch):
        x, y_true = batch
        x = x.to(self.device)
        y_true = y_true.to(self.device)
        # g = self.ds_bundle.grouper.metadata_to_group(metadata).to(self.device)
        g = ""
        # metadata = metadata.to(self.device)
        outputs = self.model(x)
        # print(outputs.shape)
        results = {
            'g': g,
            'y_true': y_true,
            'y_pred': outputs,
            'metadata': "",
        }
        return results
    
    def transfer_fit(self, flr=0):
        self.init_train()
        vgg.to(self.device)
        decoder.to(self.device)
        for e in range(self.local_epochs):
            batch_id = 0
            for batch in tqdm(self.dataloader):
                self.optimizer.zero_grad(set_to_none=True)
                images, labels, metadata = batch
                
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                new_data = []
                new_labels = []
                for style_stat in self.style_bank:
                    mean_avg, std_avg = style_stat
                    mean_avg, std_avg = torch.Tensor(mean_avg).to(self.device), torch.Tensor(std_avg).to(self.device)
                    style_stat = (mean_avg, std_avg)
                    
                    transfered_imgs = self.style_stransfer_batch(batch, alpha=0.5, style_stat=style_stat)
                    transfered_imgs.requires_grad_()
                    
                    new_data.append(transfered_imgs.cpu())
                    new_labels.append(labels.cpu())
                    
                new_data = torch.cat(new_data, dim=0)
                new_labels = torch.cat(new_labels, dim=0).to(self.device)
                
                outputs = self.model(images)
                transferred_outputs = self.model(new_data.to(self.device))
                
                loss_CE = self.ds_bundle.loss.compute(outputs, labels, return_dict=False).mean()
                loss_CE_2 = self.ds_bundle.loss.compute(transferred_outputs, new_labels, return_dict=False).mean()
                
                loss = loss_CE + loss_CE_2
                
                # self.step(loss, flr)
                self.step_loss(loss, flr)
                del new_data, new_labels
                batch_id += 1
        self.end_train()
                
    def abstract_style(self):
        vgg.to(self.device)
        modules = list(vgg.children())[:-1]
        feature_extractor = nn.Sequential(*modules)
        feature_extractor.eval()
        print(colored(f"Abstracting information of client {self.client_id}", "blue"))
        decoder.to(self.device)
        
        all_feat_sum, all_feat_square_sum, all_count, img_count = 0, 0, 0, 0
        data_loader = self.dataloader
        for it, (data, _, _) in enumerate(data_loader):
            data = data.to(self.device)
            feat = feature_extractor(data) # torch.Size([4, 512, 64, 64])
            img_count += len(data)
            del data

            feat_sum, feat_square_sum, count = calc_sum(feat)
            del feat
            
            all_feat_sum += feat_sum
            all_feat_square_sum += feat_square_sum
            all_count += count
            print(f"{it}/{len(data_loader)}")

        # import pdb; pdb.set_trace()
        feat_mean = all_feat_sum / float(all_count)
        feat_var = all_feat_square_sum / float(all_count) - feat_mean ** 2
        feat_std = torch.sqrt(feat_var + 1e-5)
            
        # del all_feat_list
        return feat_mean, feat_std
        
        