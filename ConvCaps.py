import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from capsule_utils import squash, sigmoid
from capsule_utils import dynamic_routing, em_routing

class ConvCaps(nn.Module):
    def __init__(self, opt, in_dim, out_dim, kernel=3, stride=2):
        super(ConvCaps, self).__init__()
        self.opt = opt
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.pose_len = opt.pose**2
        self.kernel = kernel
        self.stride = stride
        # Learnable Variable
        self.trans_mat = nn.Parameter(torch.randn(1, 1, kernel, kernel, in_dim, out_dim, opt.pose, opt.pose))
        self.beta_u = nn.Parameter(torch.randn(1))
        self.beta_a = nn.Parameter(torch.randn(1))

    def forward(self, capsules):
        batch_size = capsules.size(0)
        H = capsules.size(1)
        W = capsules.size(2)

        h = (H-self.kernel+1)//self.stride
        w = (W-self.kernel+1)//self.stride
        # extract patches
        patches = torch.stack([capsules[:,self.stride*i:self.stride*i+self.kernel,
                       self.stride*j:self.stride*j+self.kernel,:,:] for i in range(h) for j in range(w)], dim=1)
        poses = patches[:,:,:,:,:,:-1]
        acts = patches[:,:,:,:,:,-1]
        poses = poses.view(batch_size, h*w, self.kernel, self.kernel, \
                             self.in_dim, 1, self.opt.pose, self.opt.pose)
        # Voting from Layer L to Layer L+1
        votes = torch.matmul(poses, self.trans_mat)

        votes = votes.view(batch_size, h*w, self.kernel, self.kernel, \
                            self.in_dim, self.out_dim, self.pose_len) # N, M, K, K, I, O, 16
        acts = acts.view(batch_size, h*w, self.kernel, self.kernel, \
                            self.in_dim, 1, 1).repeat(1, 1, 1, 1, 1, self.out_dim, 1) # N, M, K, K, I, O, 1
        # Routing Process Started
        poses = []
        a_s = []
        for i in range(h*w):
            # for votes within every patches
            V = votes[:,i,:,:,:,:,:].view(batch_size, self.kernel*self.kernel*self.in_dim, self.out_dim, self.pose_len)
            a = acts[:,i,:,:,:,:,:].view(batch_size, self.kernel*self.kernel*self.in_dim, self.out_dim, 1)
            pose, act = em_routing(V, a, self.beta_u, self.beta_a, self.opt)
            poses.append(pose)
            a_s.append(act)
            
        poses = torch.cat(poses, dim=1)
        acts = torch.cat(a_s, dim=1)
        capsules = torch.cat([poses.view(batch_size, h, w, self.out_dim, self.pose_len),\
                             acts.view(batch_size, h, w, self.out_dim, 1)],dim=-1)
        return capsules
