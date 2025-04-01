import torch.nn as nn
import torchvision.transforms as transforms
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.metrics.loss import ElementwiseLoss

from .splitter import *
from .models import ResNet, Classifier, CNN, DenseNet

class ObjBundle(object):
    def __init__(self, dataset, feature_dimension, probabilistic=False) -> None:
        self.dataset = dataset
        self.feature_dimension = feature_dimension
        self.probabilistic = probabilistic
        self.input_shape = self._input_shape
        self.groupby_fields = self._domain_fields
        self.grouper = CombinatorialGrouper(dataset=dataset, groupby_fields=self.groupby_fields)
        self.loss = self._loss()
        self.train_transform = self._train_transform
        self.test_transform = self._test_transform
        self.featurizer = ResNet(self.input_shape, self.feature_dimension, probabilistic=probabilistic)
        self.classifier = Classifier(self.featurizer.n_outputs, self.n_classes)

    @property
    def is_classification(self):
        return True

    @property
    def _train_transform(self):
        raise NotImplementedError

    @property
    def _test_transform(self):
        raise NotImplementedError

    def _loss(self):
        raise NotImplementedError

    @property
    def _input_shape(self):
        return None
    
    @property
    def _domain_fields(self):
        return None

    @property
    def n_classes(self):
        return self.dataset.n_classes

#### Doesn't require re-implemented by derived classes ####
    @property
    def in_channel(self):
        return self._input_shape[0]

    @property
    def name(self):
        return self.__class__.__name__.lower()

    @property
    def key_metric(self):
        raise NotImplementedError

class IWildCam(ObjBundle):
    @property
    def _train_transform(self):
        return transforms.Compose([
            transforms.Resize((self._input_shape[1],self._input_shape[2])),
            transforms.RandomResizedCrop(self._input_shape[1], scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @property
    def _test_transform(self):
        return transforms.Compose([
            transforms.Resize((self._input_shape[1], self._input_shape[2])), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _loss(self):
        return ElementwiseLoss(loss_fn=nn.CrossEntropyLoss(reduction='none', ignore_index=-100))
        

    @property
    def _input_shape(self):
        return (3, 224, 224)
    
    @property
    def _domain_fields(self):
        return ['location',]

    @property
    def key_metric(self):
        return "acc_avg"

class PACS(ObjBundle):
    def _loss(self):
        return ElementwiseLoss(loss_fn=nn.CrossEntropyLoss(reduction='none', ignore_index=-100))
    
    @property
    def _input_shape(self):
        return (3, 224, 224)
    
    @property
    def _train_transform(self):
        return transforms.Compose([
            transforms.Resize((self._input_shape[1],self._input_shape[2])),
            transforms.RandomResizedCrop(self._input_shape[1], scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            
        ])
        
    @property
    def _test_transform(self):
        return transforms.Compose([
            transforms.Resize((self._input_shape[1], self._input_shape[2])),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @property
    def _domain_fields(self):
        return ['domain',]

    @property
    def key_metric(self):
        return "acc_avg"

class FEMNIST(ObjBundle):
    def __init__(self, dataset, feature_dimension=None, probabilistic=False) -> None:
        self.dataset = dataset
        self.probabilistic = probabilistic
        self.feature_dimension = feature_dimension
        self.input_shape = self._input_shape
        self.groupby_fields = self._domain_fields
        self.grouper = CombinatorialGrouper(dataset=dataset, groupby_fields=self.groupby_fields)
        self.loss = self._loss()
        self.train_transform = self._train_transform
        self.test_transform = self._test_transform
        self.featurizer = CNN(self.input_shape, probabilistic=probabilistic)
        self.classifier = Classifier(self.featurizer.n_outputs, dataset.n_classes)


    def _loss(self):
        return ElementwiseLoss(loss_fn=nn.CrossEntropyLoss(reduction='none', ignore_index=-100))
    
    @property
    def _input_shape(self):
        return (1, 28, 28)
    
    @property
    def _train_transform(self):
        return transforms.Compose([
            transforms.Resize((self._input_shape[1],self._input_shape[2])),
            transforms.ToTensor()
        ])

    @property
    def _test_transform(self):
        return transforms.Compose([
            transforms.Resize((self._input_shape[1], self._input_shape[2])),
            transforms.ToTensor()
        ])

    @property
    def _domain_fields(self):
        return ['domain',]
    
    @property
    def key_metric(self):
        return "acc_avg"

class OfficeHome(PACS):
    def __init__(self, dataset, feature_dimension, probabilistic=False) -> None:
        super().__init__(dataset, feature_dimension, probabilistic)
    
    @property
    def _train_transform(self):
        return transforms.Compose(
            [transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    @property
    def _test_transform(self):
        return transforms.Compose(
            [transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
    def _loss(self):
        return ElementwiseLoss(loss_fn=nn.CrossEntropyLoss(reduction='none', ignore_index=-100))
    
    @property
    def _input_shape(self):
        return (3, 224, 224)

    @property
    def _domain_fields(self):
        return ['domain',]

    @property
    def key_metric(self):
        return "acc_avg"
        
    @property
    def _input_shape(self):
        return (3, 96, 96)
    
    @property
    def _train_transform(self):
        return transforms.Compose([
            transforms.RandomResizedCrop(
                (self._input_shape[1],self._input_shape[2]),
                scale=(0.7, 1.0),
                ratio=(1.0, 1.3333333333333333),
                interpolation=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    @property
    def _test_transform(self):
        return transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize((self._input_shape[1],self._input_shape[2])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    @property
    def _domain_fields(self):
        return ['male', 'y']

    @property
    def key_metric(self):
        return "acc_wg"


    def __init__(self, dataset, feature_dimension, probabilistic=False) -> None:
        super().__init__(dataset, feature_dimension, probabilistic)
        self.featurizer = DenseNet(self.input_shape, self.feature_dimension, probabilistic=probabilistic)
        self.classifier = Classifier(self.featurizer.n_outputs, self.n_classes)

    @property
    def _input_shape(self):
        return (3, 96, 96)
    
    @property
    def _train_transform(self):
        return transforms.Compose([
            transforms.Resize((self._input_shape[1],self._input_shape[2])),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor()
            
        ])
        
    @property
    def _test_transform(self):
        return transforms.Compose([
            transforms.Resize((self._input_shape[1], self._input_shape[2])),
            transforms.ToTensor()
        ])

    @property
    def _domain_fields(self):
        return ['hospital']

    @property
    def key_metric(self):
        return "acc_avg"