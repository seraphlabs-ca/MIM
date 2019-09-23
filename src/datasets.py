"""
Toy datasets.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from PIL import Image, ImageDraw, ImageFont
from sklearn.decomposition import PCA
import torch
import torch.utils.data
import torchvision
from torchvision import datasets, transforms

import auxiliary as aux

#=============================================================================#
# Datasets
#=============================================================================#


class DistDataset(torch.utils.data.Dataset):
    """
    Wraps a Distribution with a Dataset
    """

    def __init__(self, dist, n=10000, with_index=False):
        super().__init__()

        self.dist = dist
        # dataset "size"
        self.n = n
        self.with_index = with_index

    def __call__(self, *args, **kwargs):
        return self.dist(*args, **kwargs)

    def __getitem__(self, index):
        if self.with_index:
            X, y = self.dist.sample(with_index=True)
            return X, y.flatten()
        else:
            return self.dist.sample(), 0

    def __len__(self):
        return self.n


class PCAFashionMNIST(torchvision.datasets.FashionMNIST):
    """
    MNIST with PCA dimensionality reduction
    """

    def __init__(self, k, root, train=True, transform=None, target_transform=None, download=False):
        if train:
            transform = torchvision.transforms.Compose([
                transforms.RandomApply([transforms.RandomAffine(
                    degrees=10.0,
                    translate=(0.1, 0.1),
                    scale=(0.8, 1.2),
                    shear=5.0,
                    resample=False
                )], p=0.5),
                # will scale [0, 1]
                transforms.ToTensor(),
                # dequantize
                transforms.Lambda(lambda im: ((255.0 * im + torch.rand_like(im)) / 256.0).clamp(1e-3, 1 - 1e-3)),
            ])
        else:
            # will scale [0, 1]
            transform = transforms.ToTensor()

        super().__init__(
            root=root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

        self.k = k

        if self.train:
            data = self.train_data
        else:
            data = self.test_data

        X = data.type(torch.float32).view((data.shape[0], -1)) / data.max()
        self.pca = PCA(n_components=k)
        self.pca.fit(X)

        print("k = {k} captures {var:.3f} variance of dataset".format(
            k=self.k, var=self.pca.explained_variance_ratio_.sum()))

        # X_recon = self.pca.inverse_transform(self.pca.transform(X[:64]))
        # torchvision.utils.save_image(torch.tensor(X_recon[:64, :]).view((-1, 1, 28, 28)), "recon.png", normalize=True)
        # torchvision.utils.save_image(X[:64, :].view((-1, 1, 28, 28)), "real.png", normalize=True)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            data = self.train_data
            targets = self.train_labels
        else:
            data = self.test_data
            targets = self.test_labels

        img, target = data[index], int(targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # project to PCA coef
        coef = torch.tensor(self.pca.transform(img.view((1, -1)))).type(torch.float32)

        return coef, target
