import argparse
import torch
from torchvision import datasets, transforms

'''TODO: Add SmallNORB datasets'''

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
    # training configuration
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-lr', type=float, default=1e-3)
    parser.add_argument('-epochs', type=int, default=10)
    parser.add_argument('-r', type=int, default=3)
    parser.add_argument('-use_cuda', default=True)
    parser.add_argument('-print_every', type=int, default=10)
    # model configurarion
    parser.add_argument('-img_size', type=int, default=28)
    parser.add_argument('-num_class', type=int, default=10)
    parser.add_argument('-in_channel', type=int, default=1)
    parser.add_argument('-pose', type=int, default=4)
    # num feature map
    parser.add_argument('-A', type=int, default=8)
    parser.add_argument('-B', type=int, default=8)
    parser.add_argument('-C', type=int, default=8)
    parser.add_argument('-D', type=int, default=8)
    # kernel size
    parser.add_argument('-conv1_ksize', type=int, default=5)
    parser.add_argument('-convcaps1_ksize', type=int, default=3)
    parser.add_argument('-convcaps2_ksize', type=int, default=3)
    # stride
    parser.add_argument('-conv1_stride', type=int, default=2)
    parser.add_argument('-convcaps1_stride', type=int, default=2)
    parser.add_argument('-convcaps2_stride', type=int, default=1)
    # routing config
    parser.add_argument('-iter', type=int, default=3)
    parser.add_argument('-lambda_', type=float, default=0.1)
    # loss
    parser.add_argument('-loss', type=str, default='spread_loss')
    parser.add_argument('-alpha', type=float, default=5e-4)
    parser.add_argument('-margin', type=float, default=0.2)

    opt = parser.parse_args()

    return opt

if __name__ == '__main__':
    opt = get_opts()
    get_dataloader(opt)
