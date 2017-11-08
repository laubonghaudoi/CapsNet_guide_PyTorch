import argparse
import torch
from torchvision import datasets, transforms


def get_dataloader(opt):
    # MNIST Dataset
    train_dataset = datasets.MNIST(root='./data/',
                                   train=True,
                                   transform=transforms.ToTensor(),
                                   download=True)

    test_dataset = datasets.MNIST(root='./data/',
                                  train=False,
                                  transform=transforms.ToTensor())

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=opt.batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=opt.batch_size,
                                              shuffle=True)

    return train_loader, test_loader


def get_opts():
    parser = argparse.ArgumentParser(description='CapsNet')

    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-lr', type=float, default=1e-5)
    parser.add_argument('-epochs', type=int, default=10)
    parser.add_argument('-r', type=int, default=3)
    parser.add_argument('-use_cuda', default=True)
    parser.add_argument('-print_every', type=int, default=10)
    opt = parser.parse_args()

    return opt
