import torch
import torch.cuda
from torch.utils.data import RandomSampler, Subset
from wilds.common.data_loaders import get_eval_loader
from wilds import get_dataset

import os
import time
import argparse
import json

from src.server import *
from src.client import *
from src.splitter import *
from src.utils import *
from src.dataset_bundle import *
import src.datasets as my_datasets

import wandb
from wandb_env import WANDB_ENTITY, WANDB_PROJECT

import warnings
warnings.filterwarnings("ignore")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:20000"

"""
The main file function:
1. Load the hyperparameter dict.
2. Init Wandb
3. Initialize data (preprocess, data splits, etc.)
4. Initialize clients. 
5. Initialize Server.
6. Register clients at the server.
7. Start the server.
"""
def main(args):
    hparam = vars(args)
    config_file = args.config_file
    with open(config_file) as fh:
        config = json.load(fh)
    hparam.update(config)
    # setup WanDB
    wandb.init(project=hparam['group'],
                entity=WANDB_ENTITY,
                name=hparam['wandb_instance'],
                config=hparam)
    wandb.run.log_code()
    config['id'] = wandb.run.id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = hparam['seed']
    set_seed(seed)
    data_path = hparam['data_path']
    print(f"data_path: {data_path}")
    if not os.path.exists(data_path + "opt_dict/"): os.makedirs(data_path + "opt_dict/")
    if not os.path.exists(data_path + "models/"): os.makedirs(data_path + "models/")

    # optimizer preprocess
    if hparam['optimizer'] == 'torch.optim.SGD':
        hparam['optimizer_config'] = {'lr':hparam['lr'], 'momentum': hparam['momentum'], 'weight_decay': hparam['weight_decay']}
    elif hparam['optimizer'] == 'torch.optim.Adam' or hparam['optimizer'] == 'torch.optim.AdamW':
        hparam['optimizer_config'] = {'lr':hparam['lr'], 'eps': hparam['eps'], 'weight_decay': hparam['weight_decay']}
    print(f"hparam: {hparam}")
    # initialize data
    if hparam['dataset'].lower() == 'pacs':
        dataset = my_datasets.PACS(version='1.0', root_dir=hparam['dataset_path'], split_scheme = hparam["split_scheme"], download=True)
        print("downloading...")
    elif hparam['dataset'].lower() == 'officehome':
        dataset = my_datasets.OfficeHome(version='1.0', root_dir=hparam['dataset_path'], download=True, split_scheme=hparam["split_scheme"])
    elif hparam['dataset'].lower() == 'femnist':
        dataset = my_datasets.FEMNIST(version='1.0', root_dir=hparam['dataset_path'], download=True)
    else:
        dataset = get_dataset(dataset=hparam["dataset"].lower(), root_dir=hparam['dataset_path'], download=True)
        
    if hparam['client_method'] == "FedSR":
        ds_bundle = eval(hparam["dataset"])(dataset, hparam["feature_dimension"], probabilistic=True)
    else:
        ds_bundle = eval(hparam["dataset"])(dataset, hparam["feature_dimension"], probabilistic=False)

    total_subset = dataset.get_subset('train', transform=ds_bundle.train_transform)
    testloader = {}
    
    for split in dataset.split_names:
        if split != 'train':
            ds = dataset.get_subset(split, transform=ds_bundle.test_transform)
            dl = get_eval_loader(loader='standard', dataset=ds, batch_size=hparam["test_batch_size"])
            testloader[split] = dl
    
    sampler = RandomSampler(total_subset, replacement=True)
    subset_size = 1024 # Need to change
    subset_indices = torch.randperm(len(total_subset))[:subset_size]
    val_subset = Subset(total_subset, subset_indices)
    val_sampler = RandomSampler(val_subset, replacement=True)
    global_dataloader = DataLoader(total_subset, batch_size=hparam["batch_size"], sampler=sampler, num_workers=32)
    val_dataloader = DataLoader(val_subset, batch_size=256, sampler=val_sampler, num_workers=32)

    num_shards = hparam['num_clients']
    if num_shards == 1:
        training_datasets = [total_subset]
    elif num_shards > 1:
        print(f"splitting...")
        training_datasets = NonIIDSplitter(num_shards=num_shards, iid=hparam['iid'], seed=seed).split(dataset.get_subset('train'), ds_bundle.groupby_fields, transform=ds_bundle.train_transform, dataset_name=hparam["dataset"])
    else:
        raise ValueError("num_shards should be greater or equal to 1, we got {}".format(num_shards))

    # initialize client
    clients = []
    for k in tqdm(range(hparam["num_clients"]), leave=False):
        if hparam['client_method'] == "DGClient":
            client = eval("DGClient")(k, device, training_datasets[k], ds_bundle, hparam)
        else:
            client = eval(hparam["client_method"])(k, device, training_datasets[k], ds_bundle, hparam)
        clients.append(client)
    # del message; gc.collect() 

    # initialize server (model should be initialized in the server. )
    if hparam["server_method"] == "DGServer":
        central_server = eval(hparam["server_method"])(device, ds_bundle, val_dataloader, 
                                                       hparam)
    else:
        central_server = eval(hparam["server_method"])(device, ds_bundle, hparam)

    if hparam['start_epoch'] == 0:
        central_server.setup_model(None, 0)
    else:
        central_server.setup_model(hparam['resume_file'], hparam['start_epoch'])
        
    central_server.register_clients(clients)
    central_server.register_testloader(testloader)
    # do federated learning
    central_server.fit()
    
    # bye!
    message = "...done all learning process!\n...exit program!"
    logging.info(message)
    time.sleep(3)
    exit()

if __name__ == "__main__":
    def bool_string(s):
        if s not in {'False', 'True'}:
            raise ValueError('Not a valid boolean string')
        return s == 'True'

    parser = argparse.ArgumentParser(description='FedDG Benchmark')
    parser.add_argument('--config_file', help='config file', default="config.json")
    parser.add_argument('--seed', default=1001, type=int)
    parser.add_argument('--num_clients', default=1, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--test_batch_size', default=64, type=int)
    parser.add_argument('--iid', default=1, type=float)
    parser.add_argument('--server_method', default='FedAvg')
    parser.add_argument('--fixed_meta', type=bool_string, default=False)
    parser.add_argument('--val', type=bool_string, default=False)
    parser.add_argument('--partial_data', type=bool_string, default=False, help='select only a portion of data to calculate style')
    parser.add_argument('--fraction', default=1, type=float)
    parser.add_argument('--f', default=10, type=int)
    parser.add_argument('--num_rounds', default=20, type=int)
    parser.add_argument('--output_size', default=224, type=int, help='output size for the training data')
    parser.add_argument('--dataset', default='PACS')
    parser.add_argument('--split_scheme', default='official')
    parser.add_argument('--client_method', default='ERM')
    parser.add_argument('--local_epochs', default=1, type=int)
    parser.add_argument('--n_groups_per_batch', default=2, type=int)
    parser.add_argument('--optimizer', default='torch.optim.Adam')
    parser.add_argument('--feature_dimension', default=2048, type=int)
    parser.add_argument('--lr', default=3e-5, type=float)
    parser.add_argument('--momentum', default=0, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--param_1', default=1.0, type=float)
    parser.add_argument('--hparam1', default=1, type=float, help="irm: lambda; rex: lambda; fish: meta_lr; mixup: alpha; mmd: lambda; coral: lambda; groupdro: groupdro_eta; fedprox: mu; feddg: ratio; fedadg: alpha; fedgma: mask_threshold; fedsr: l2_regularizer;")
    parser.add_argument('--hparam2', default=1, type=float, help="fedsr: cmi_regularizer; irm: penalty_anneal_iters;fedadg: second_local_epochs")
    parser.add_argument('--infoNCET', type=float, default=0.02, help='The InfoNCE temperature, for prototype only')
    args = parser.parse_args()
    main(args)

