import torch
import torch.nn as nn
import torch.nn.functional as F


class PrimaryCaps(nn.Module):
    '''
    The `PrimaryCaps` layer consists of 32 capsule units. Each unit takes
    the output of the `Conv1` layer, which is a `[256, 20, 20]` feature
    tensor (ignoring `batch_size`), and performs a 2D convolution with 8
    output channels, kernel size 9 and stride 2, thus outputing a [8, 6, 6]
    tensor. In other words, you can see these 32 capsules as 32 paralleled 2D
    convolutional layers. Then we concatenate these 32 capsules' outputs and
    flatten them into a tensor of size `[1152, 8]`, representing 1152 8D
    vectors, and send it to the next layer `DigitCaps`.

    As indicated in Section 4, Page 4 in the paper, *One can see PrimaryCaps
    as a Convolution layer with Eq.1 as its block non-linearity.*, outputs of
    the `PrimaryCaps` layer are squashed before passing to the next layer.

    Reference: Section 4, Fig. 1
    '''

    def __init__(self):
        '''
        We build 8 capsule units in the `PrimaryCaps` layer, each can be
        seen as a 2D convolution layer.
        '''
        super(PrimaryCaps, self).__init__()

        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels=256,
                      out_channels=8,
                      kernel_size=9,
                      stride=2)
            for i in range(32)
        ])

    def forward(self, x):
        '''
        Each capsule outputs a [batch_size, 8, 6, 6] tensor, we need to
        flatten and concatenate them into a [batch_size, 8, 6*6, 32] size
        tensor and flatten and transpose into `u` [batch_size, 1152, 8], 
        where each [batch_size, 1152, 1] size tensor is the `u_i` in Eq.2. 

        #### Dimension transformation in this layer(ignoring `batch_size`):
        [256, 20, 20] --> [8, 6, 6] x 32 capsules --> [1152, 8]

        Note: `u_i` is one [1, 8] in the final [1152, 8] output, thus there are
        1152 `u_i`s.
        '''
        batch_size = x.size(0)

        u = []
        for i in range(32):
            # Input: [batch_size, 256, 20, 20]
            assert x.data.size() == torch.Size([batch_size, 256, 20, 20])

            u_i = self.capsules[i](x)
            assert u_i.size() == torch.Size([batch_size, 8, 6, 6])
            # u_i: [batch_size, 8, 6, 6]
            u_i = u_i.view(batch_size, 8, -1, 1)
            # u_i: [batch_size, 8, 36]
            u.append(u_i)

        # u: [batch_size, 8, 36, 1] x 32
        u = torch.cat(u, dim=3)
        # u: [batch_size, 8, 36, 32]
        u = u.view(batch_size, 8, -1)
        # u: [batch_size, 8, 1152]
        u = torch.transpose(u, 1, 2)
        # u: [batch_size, 1152, 8]
        assert u.data.size() == torch.Size([batch_size, 1152, 8])

        # Squash before output
        u_squashed = self.squash(u)

        return u_squashed

    def squash(self, u):
        '''
        Args:
            `u`: [batch_size, 1152, 8]

        Return:
            `u_squashed`: [batch_size, 1152, 8]

        In CapsNet, we use the squash function after the output of both 
        capsule layers. Squash functions can be seen as activating functions
        like sigmoid, but for capsule layers rather than traditional fully
        connected layers, as they squash vectors instead of scalars.

        v_j = (norm(s_j) ^ 2 / (1 + norm(s_j) ^ 2)) * (s_j / norm(s_j))

        Reference: Eq.1 in Section 2.
        '''
        batch_size = u.size(0)

        # u: [batch_size, 1152, 8]
        square = u ** 2

        # square_sum for u: [batch_size, 1152]
        square_sum = torch.sum(square, dim=2)

        # norm for u: [batch_size, 1152]
        norm = torch.sqrt(square_sum)

        # factor for u: [batch_size, 1152]
        factor = norm ** 2 / (norm * (1 + norm ** 2))

        # u_squashed: [batch_size, 1152, 8]
        u_squashed = factor.unsqueeze(2) * u
        assert u_squashed.size() == torch.Size([batch_size, 1152, 8])

        return u_squashed
