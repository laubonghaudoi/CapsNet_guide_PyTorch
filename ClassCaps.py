import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from capsule_utils import squash, sigmoid
from capsule_utils import dynamic_routing, em_routing

class ClassCaps(nn.Module):
    def __init__(self, opt, in_dim, out_dim):
        super(ClassCaps, self).__init__()
        self.opt = opt
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.pose_len = opt.pose**2
        # Learnable Variable
        self.trans_mat = nn.Parameter(torch.randn(1, 1, in_dim, out_dim, 4, 4))
        self.beta_u = nn.Parameter(torch.randn(1))
        self.beta_a = nn.Parameter(torch.randn(1))

    def forward(self, capsules):
        batch_size = capsules.size(0)
        H = capsules.size(1)
        W = capsules.size(2)

        poses = capsules[:,:,:,:,:-1]
        acts = capsules[:,:,:,:,-1]
        poses = poses.view(batch_size, H*W, self.in_dim, 1, self.opt.pose, self.opt.pose)
        votes = torch.matmul(poses, self.trans_mat)

        votes = votes.view(batch_size, H*W, self.in_dim, self.out_dim, self.pose_len)
        acts = acts.view(batch_size, H*W, self.in_dim, 1, 1).repeat(1, 1, 1, self.out_dim, 1)
        # Coordination Addition
        add = []
        for x in range(H):
            for y in range(W):
                add.append([1.*x/H, 1.*y/W]) # normalization
        add = Variable(torch.Tensor(add)).view(1,H*W,1,1,2)
        if self.opt.use_cuda & torch.cuda.is_available():
            add = add.cuda()
        votes[:,:,:,:,:2] = votes[:,:,:,:,:2] + add

        # Routing Process Started
        V = votes.view(batch_size, H*W*self.in_dim, self.out_dim, self.pose_len)
        a = acts.view(batch_size, H*W*self.in_dim, self.out_dim, 1)
        poses, acts = em_routing(V, a, self.beta_u, self.beta_a, self.opt)
        capsules = torch.cat([poses.view(batch_size, self.out_dim, self.pose_len),\
                             acts.view(batch_size, self.out_dim, 1)],dim=-1)
        return capsules
