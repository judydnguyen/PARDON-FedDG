import torch
import torch.nn as nn
import torchvision.models
from torch.nn.functional import avg_pool2d

def remove_batch_norm_from_resnet(model):
    fuse = torch.nn.utils.fusion.fuse_conv_bn_eval
    model.eval()

    model.conv1 = fuse(model.conv1, model.bn1)
    model.bn1 = Identity()

    for name, module in model.named_modules():
        if name.startswith("layer") and len(name) == 6:
            for b, bottleneck in enumerate(module):
                for name2, module2 in bottleneck.named_modules():
                    if name2.startswith("conv"):
                        bn_name = "bn" + name2[-1]
                        setattr(bottleneck, name2,
                                fuse(module2, getattr(bottleneck, bn_name)))
                        setattr(bottleneck, bn_name, Identity())
                if isinstance(bottleneck.downsample, torch.nn.Sequential):
                    bottleneck.downsample[0] = fuse(bottleneck.downsample[0],
                                                    bottleneck.downsample[1])
                    bottleneck.downsample[1] = Identity()
    model.train()
    return model


class Identity(nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class CNN(nn.Module):
    def __init__(self, input_shape, probabilistic=False):
        super(CNN,self).__init__()
        self.n_outputs = 2048
        self.probabilistic = probabilistic
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0],out_channels=16,kernel_size=5,padding=2),  # in_channels, out_channels, kernel_size
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),  # kernel_size, stride
            nn.Conv2d(in_channels=16,out_channels=64,kernel_size=5,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        if self.probabilistic:
            self.fc = nn.Linear(in_features=7*7*64,out_features=self.n_outputs * 2)
        else:
            self.fc = nn.Linear(in_features=7*7*64,out_features=self.n_outputs)
    def forward(self,x):
        feature=self.fc(self.conv(x).view(x.shape[0], -1))
        return feature

class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""
    def __init__(self, input_shape, feature_dimension=2048, probabilistic=False):
        super(ResNet, self).__init__()
        self.probabilistic = probabilistic
        self.network = torchvision.models.resnet50(pretrained=True)
        self.n_outputs = feature_dimension

        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False)

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]
        self.dropout = nn.Dropout(0)
        if probabilistic:
            self.network.fc = nn.Linear(self.network.fc.in_features,self.n_outputs*2)
        else:
            self.network.fc = nn.Linear(self.network.fc.in_features,self.n_outputs)
        self._internal_features = nn.Sequential(self.network.conv1,
                                       self.network.bn1,
                                       nn.ReLU(),
                                       self.network.layer1,
                                       self.network.layer2,
                                       self.network.layer3,
                                       self.network.layer4)

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.dropout(self.network(x))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    
    # def extract_features(self, x):
    #     """
    #     Extract intermediate features (before the fc layer) from the model.
    #     """
    #     # Pass the input through the convolutional layers
    #     x = self.network.conv1(x)
    #     x = self.network.bn1(x)
    #     x = self.network.relu(x)
    #     x = self.network.maxpool(x)

    #     x = self.network.layer1(x)
    #     x = self.network.layer2(x)
    #     x = self.network.layer3(x)
    #     x = self.network.layer4(x)

    #     return x

    def features(self, x: torch.Tensor) -> torch.Tensor:
        out = self._internal_features(x)
        out = avg_pool2d(out, out.shape[2])
        feat = out.view(out.size(0), -1)
        return feat
    
class DenseNet(torch.nn.Module):
    def __init__(self, input_shape, feature_dimension=2048, probabilistic=False, pretrained=True):
        super(DenseNet, self).__init__()
        self.probabilistic = probabilistic
        self.network = torchvision.models.densenet121(pretrained=pretrained)
        self.n_outputs = feature_dimension

        # self.network = remove_batch_norm_from_resnet(self.network)

        # adapt number of channels
        nc = input_shape[0]
        self.dropout = nn.Dropout(0)
        if probabilistic:
            self.network.classifier = nn.Linear(self.network.classifier.in_features,self.n_outputs*2)
        else:
            self.network.classifier = nn.Linear(self.network.classifier.in_features,self.n_outputs)

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.dropout(self.network(x))

    # def train(self, mode=True):
    #     """
    #     Override the default train() to freeze the BN parameters
    #     """
    #     super().train(mode)
    #     self.freeze_bn()

    # def freeze_bn(self):
    #     for m in self.network.modules():
    #         if isinstance(m, nn.BatchNorm2d):
    #             m.eval()

def Classifier(in_features, out_features, is_nonlinear=False):
    if is_nonlinear:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 2, in_features // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 4, out_features))
    else:
        return torch.nn.Linear(in_features, out_features)
