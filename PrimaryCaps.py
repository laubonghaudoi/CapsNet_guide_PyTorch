import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from capsule_utils import squash, sigmoid
from capsule_utils import dynamic_routing, em_routing

class PrimaryCaps(nn.Module):
    def __init__(self, opt, in_dim, out_dim):
        super(PrimaryCaps, self).__init__()
        self.opt = opt
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.pose_len = opt.pose**2

    def forward(self, x):
        x = x.permute(0,2,3,1)
        batch_size = x.size(0)
        H = x.size(1)
        W = x.size(2)

        self.trans_mat = nn.Parameter(torch.randn(1, H, W, self.in_dim, self.out_dim*(self.pose_len+1)))

        x = x.view(batch_size, H, W, 1, self.in_dim)
        # transformation
        x = torch.matmul(x, self.trans_mat)
        # capsule = pose + activation
        capsules = x.view(batch_size, H, W, self.out_dim, (self.pose_len+1))
        return capsules
