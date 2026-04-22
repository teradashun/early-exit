import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset


def get_datasets(batch_size, data_name):
    if data_name == "MNIST":
        train_dataset = torchvision.datasets.MNIST(root="./data",
                                            train=True,
                                            transform=transforms.ToTensor(),
                                            download=True)
        
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=False)

        test_dataset = torchvision.datasets.MNIST(root="./data",
                                            train=False,
                                            transform=transforms.ToTensor(),
                                            download=True)

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False)
    
    elif data_name == "CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10(root="./data/CIFAR_10",
                                            train=True,
                                            transform=transforms.ToTensor(),
                                            download=True)
        
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=False)

        test_dataset = torchvision.datasets.CIFAR10(root="./data/CIFAR_10",
                                            train=False,
                                            transform=transforms.ToTensor(),
                                            download=True)

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False)

    return train_loader, test_loader


def split_dataset(dataset, num_clients, dirichlet_alpha):
    num_classes = 10
    class_indices = {i: [] for i in range(num_classes)}

    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)
    
    for indices in class_indices.values():
        np.random.shuffle(indices)
    
    client_indices = {i: [] for i in range(num_clients)}

    for indices in class_indices.values():
        proportions = np.random.dirichlet(np.repeat(dirichlet_alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
        split_indices = np.split(indices, proportions)

        for i, idx in enumerate(split_indices):
            client_indices[i].extend(idx)
    
    subsets = [Subset(dataset, client_indices[i]) for i in range(num_clients)]
    return subsets