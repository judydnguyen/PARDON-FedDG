from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import pandas as pd
from PIL import Image
from pathlib import Path

from wilds.common.metrics.all_metrics import Accuracy
from wilds.common.grouper import CombinatorialGrouper
from wilds.datasets.wilds_dataset import WILDSDataset

class PACS(WILDSDataset):
    _dataset_name = "pacs"
    _versions_dict = {
        '1.0': {
            "download_url": "https://drive.google.com/uc?id=1JFr8f805nMUelQWWmfnJR3y4_SYoN5Pd",
            "compressed_size": "174_167_459"
            }
    }
    def __init__(
        self, 
        version: str = None,
        root_dir: str = "data",
        download: bool = False,
        split_scheme: str = "official"
    ):
        # Dataset information
        self._version: Optional[str] = version
        self._split_scheme: str = split_scheme
        self._original_resolution = (224, 224)
        self._y_type: str = "long"
        self._y_size: int = 1
        # Path of the dataset
        self._data_dir: str = Path(self.initialize_data_dir(root_dir, download))
        print(self._data_dir)
        # The original dataset contains 7 categories. 
        if self._split_scheme == 'official':
            metadata_filename = "metadata.csv"
            print('dcc')
        else:
            print('acc')
            metadata_filename = "{}.csv".format(self._split_scheme)
        self._n_classes = 7

        # Load splits
        df = pd.read_csv(self._data_dir / metadata_filename)
        # Filenames
        self._input_array = df['path'].values
        # Splits
        self._split_dict = {'train': 0, 'val': 1, 'test': 2, 'id_val': 3, 'id_test': 4}
        self._split_names = {'train': 'Train', 'val': 'Validation (OOD/Trans)',
                                'test': 'Test (OOD/Trans)', 'id_val': 'Validation (ID/Cis)',
                                'id_test': 'Test (ID/Cis)'}

        df['split_id'] = df['split'].apply(lambda x: self._split_dict[x])
        self._split_array = df["split_id"].values
        # Y
        self._y_array = torch.from_numpy(df["y"].values).type(torch.LongTensor)
        # Populate metadata fields
        self._metadata_fields = ["domain", "y", "idx"]
        self._metadata_array = torch.tensor(np.stack([df['domain_remapped'].values,
                            df['y'].values, np.arange(df['y'].shape[0])], axis=1))
        self._eval_grouper = CombinatorialGrouper(
            dataset=self,
            groupby_fields=(['domain']))

        super().__init__(root_dir, download, self._split_scheme)

    def get_input(self, idx) -> str:
        """
        Args:
            - idx (int): Index of a data point
        Output:
            - x (Tensor): Input features of the idx-th data point
        """

        # All images are in the train folder
        img_path = self.data_dir / self._input_array[idx]
        img = Image.open(img_path).convert("RGB")

        return img

    def eval(
        self,
        y_pred: torch.Tensor,
        y_true: torch.LongTensor,
        metadata: torch.Tensor,
        prediction_fn=None,
    ) -> Tuple[Dict[str, Any], str]:
        """
        Computes all evaluation metrics.
        Args:
            - y_pred (Tensor): Predictions from a model. By default, they are predicted labels (LongTensor).
                               But they can also be other model outputs such that prediction_fn(y_pred)
                               are predicted labels.
            - y_true (LongTensor): Ground-truth labels
            - metadata (Tensor): Metadata
            - prediction_fn (function): A function that turns y_pred into predicted labels
        Output:
            - results (dictionary): Dictionary of evaluation metrics
            - results_str (str): String summarizing the evaluation metrics
        """
        metric: Accuracy = Accuracy(prediction_fn=prediction_fn)
        return self.standard_eval(
            metric, y_pred, y_true
        )

class FEMNIST(WILDSDataset):
    _dataset_name = "femnist"
    _versions_dict = {
        '1.0': {
            "download_url": "https://drive.google.com/uc?id=1JFr8f805nMUelQWWmfnJR3y4_SYoN5Pd",
            "compressed_size": "174_167_459"
            }
    }
    def __init__(
        self, 
        version: str = None,
        root_dir: str = "data",
        download: bool = True,
        split_scheme: str = "official"
    ):
        # Dataset information
        self._version: Optional[str] = version
        self._split_scheme: str = split_scheme
        self._original_resolution = (128, 128)
        self._y_type: str = "long"
        self._y_size: int = 1
        # Path of the dataset
        self._data_dir: str = Path(self.initialize_data_dir(root_dir, download))

        # The original dataset contains 7 categories. 
        metadata_filename = "femnist_new_metadata.csv"
        self._n_classes = 62

        # Load splits
        df = pd.read_csv(self._data_dir / metadata_filename)
        # Filenames
        self._input_array = df['path'].values
        # Splits
        self._split_dict = {'train': 0, 'val': 1, 'test': 2, 'id_val': 3, 'id_test': 4}
        self._split_names = {'train': 'Train', 'val': 'Validation (OOD/Trans)',
                                'test': 'Test (OOD/Trans)', 'id_val': 'Validation (ID/Cis)',
                                'id_test': 'Test (ID/Cis)'}

        df['split_id'] = df['split'].apply(lambda x: self._split_dict[x])
        self._split_array = df["split_id"].values
        # Y
        self._y_array = torch.from_numpy(df["y"].values).type(torch.LongTensor)
        # Populate metadata fields
        self._metadata_fields = ["domain", "y", "idx"]
        self._metadata_array = torch.tensor(np.stack([df['domain_remapped'].values,
                            df['y'].values, np.arange(df['y'].shape[0])], axis=1))
        self._eval_grouper = CombinatorialGrouper(
            dataset=self,
            groupby_fields=(['domain']))

        super().__init__(root_dir, download, self._split_scheme)


    def get_input(self, idx) -> str:
        """
        Args:
            - idx (int): Index of a data point
        Output:
            - x (Tensor): Input features of the idx-th data point
        """

        # All images are in the train folder
        img_path = self.data_dir / self._input_array[idx]
        img = Image.open(img_path).convert("L")

        # Resize the image to 28x28 pixels
        size = (28, 28)
        resized_img = img.resize(size)
        return resized_img

    def eval(
        self,
        y_pred: torch.Tensor,
        y_true: torch.LongTensor,
        metadata: torch.Tensor,
        prediction_fn=None,
    ) -> Tuple[Dict[str, Any], str]:
        """
        Computes all evaluation metrics.
        Args:
            - y_pred (Tensor): Predictions from a model. By default, they are predicted labels (LongTensor).
                               But they can also be other model outputs such that prediction_fn(y_pred)
                               are predicted labels.
            - y_true (LongTensor): Ground-truth labels
            - metadata (Tensor): Metadata
            - prediction_fn (function): A function that turns y_pred into predicted labels
        Output:
            - results (dictionary): Dictionary of evaluation metrics
            - results_str (str): String summarizing the evaluation metrics
        """
        metric: Accuracy = Accuracy(prediction_fn=prediction_fn)
        return self.standard_eval(
            metric, y_pred, y_true
        )

class OfficeHome(WILDSDataset):
    _dataset_name = "office_home"
    _versions_dict = {
        '1.0': {
            "download_url": "https://drive.google.com/uc?id=1JFr8f805nMUelQWWmfnJR3y4_SYoN5Pd",
            "compressed_size": "174_167_459"
            }
    }
    def __init__(
        self, 
        version: str = None,
        root_dir: str = "data",
        download: bool = False,
        split_scheme: str = "official"
    ):
        # Dataset information
        self._version: Optional[str] = version
        self._split_scheme: str = split_scheme
        self._original_resolution = (224, 224)
        self._y_type: str = "long"
        self._y_size: int = 1
        # Path of the dataset
        self._data_dir: str = Path(self.initialize_data_dir(root_dir, download))

        # The original dataset contains 7 categories. 
        if self._split_scheme == 'official':
            metadata_filename = "OfficeHome_metadata.csv"
        else:
            metadata_filename = "{}.csv".format(self._split_scheme)
        self._n_classes = 65

        # Load splits
        df = pd.read_csv(self._data_dir / metadata_filename)
        # Filenames
        self._input_array = df['path'].values
        # Splits
        self._split_dict = {'train': 0, 'val': 1, 'test': 2, 'id_val': 3, 'id_test': 4}
        self._split_names = {'train': 'Train', 'val': 'Validation (OOD/Trans)',
                                'test': 'Test (OOD/Trans)', 'id_val': 'Validation (ID/Cis)',
                                'id_test': 'Test (ID/Cis)'}
        print(df['split'])
        df['split_id'] = df['split'].apply(lambda x: self._split_dict[x])
        self._split_array = df["split_id"].values
        # Y
        self._y_array = torch.from_numpy(df["y"].values).type(torch.LongTensor)
        # Populate metadata fields
        self._metadata_fields = ["domain", "y", "idx"]
        self._metadata_array = torch.tensor(np.stack([df['domain_remapped'].values,
                            df['y'].values, np.arange(df['y'].shape[0])], axis=1))
        self._eval_grouper = CombinatorialGrouper(
            dataset=self,
            groupby_fields=(['domain']))

        super().__init__(root_dir, download, self._split_scheme)


    def get_input(self, idx) -> str:
        """
        Args:
            - idx (int): Index of a data point
        Output:
            - x (Tensor): Input features of the idx-th data point
        """

        # All images are in the train folder
        img_path = self.data_dir / self._input_array[idx]
        img = Image.open(img_path).convert("RGB")

        return img

    def eval(
        self,
        y_pred: torch.Tensor,
        y_true: torch.LongTensor,
        metadata: torch.Tensor,
        prediction_fn=None,
    ) -> Tuple[Dict[str, Any], str]:
        """
        Computes all evaluation metrics.
        Args:
            - y_pred (Tensor): Predictions from a model. By default, they are predicted labels (LongTensor).
                               But they can also be other model outputs such that prediction_fn(y_pred)
                               are predicted labels.
            - y_true (LongTensor): Ground-truth labels
            - metadata (Tensor): Metadata
            - prediction_fn (function): A function that turns y_pred into predicted labels
        Output:
            - results (dictionary): Dictionary of evaluation metrics
            - results_str (str): String summarizing the evaluation metrics
        """
        metric: Accuracy = Accuracy(prediction_fn=prediction_fn)
        return self.standard_eval(
            metric, y_pred, y_true
        )
