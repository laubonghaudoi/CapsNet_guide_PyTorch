import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class DigitCaps(nn.Module):
    '''
    The `DigitCaps` layer consists of 10 16D capsules. Compared to the traditional
    scalar output neurons in fully connected layers(FCN), the `DigitCaps` layer
    can be seen as an FCN with 16-dimensional output neurons, where we call
    these neurons "capsules".

    In this layer, we take the input `[1152, 8]` tensor `u` as 1152 [8,] vectors
    `u_i`, each `u_i` is a 8D output of the capsules from `PrimaryCaps` (see Eq.2
    in Section 2, Page 2) and sent to the 10 capsules. For each capsule, the tensor
    is first transformed by `W_ij`s into [1152, 16] size. Then we perform the Dynamic
    Routing algorithm to get the output `v_j` of size [16,]. As there are 10 capsules,
    the final output is [16, 10] size.

    #### Dimension transformation in this layer(ignoring `batch_size`):
    [1152, 8] --> [1152, 16] --> [1, 16] x 10 capsules --> [10, 16] output

    Note that in our codes we have vectorized these computations, so the dimensions
    above are just for understanding, actual dimensions of tensors are different.
    '''

    def __init__(self, opt):
        '''
        There is only one parameter in this layer, `W` [1152, 10, 16, 8], where 
        every [16, 8] is a weight matrix W_ij in Eq.2, that is, there are 11520
        `W_ij`s in total.

        The the coupling coefficients `b` [1152, 10] is a temporary variable which
        does NOT belong to the layer's parameters. In other words, `b` is not updated
        by gradient back-propagations. Instead, we update `b` by Dynamic Routing
        in every forward propagation. See docstring of `self.forward` for details.
        '''
        super(DigitCaps, self).__init__()
        self.opt = opt

        self.W = nn.Parameter(torch.randn(1152, 10, 16, 8))

    def forward(self, u):
        '''
        Args:
            `u`: [batch_size, 1152, 8]
        Return:
            `v`: [batch_size, 10, 16]

        In this layer, we vectorize our computations by calling `W` and using
        `torch.matmul()`. Thus the full computaion steps are as follows.
            1. Expand `W` into batches and compute `u_hat` (Eq.2)
            2. Line 2: Initialize `b` into zeros 
            3. Line 3: Start Routing for `r` iterations:
                1. Line 4: c = softmax(b)
                2. Line 5: s = sum(c * u_hat)
                3. Line 6: v = squash(s)
                4. Line 7: b += u_hat * v

        The coupling coefficients `b` can be seen as a kind of attention matrix
        in the attentional sequence-to-sequence networks, which is widely used in
        Neural Machine Translation systems. For tutorials on  attentional seq2seq
        models, see https://arxiv.org/abs/1703.01619 or
        http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

        Reference: Section 2, Procedure 1
        '''
        batch_size = u.size(0)

        # First, we need to expand the dimensions of `W` and `u` to compute `u_hat`
        assert u.size() == torch.Size([batch_size, 1152, 8])
        # u: [batch_size, 1152, 8, 1]
        u = torch.unsqueeze(u, dim=3)
        # u_stack: [batch_size, 1152, 10, 8, 1]
        u_stack = torch.stack([u for i in range(10)], dim=2)
        # W_batch: [batch_size, 1152, 10, 16, 8]
        W_batch = torch.stack(
            [self.W for i in range(batch_size)], dim=0)

        # Now we compute u_hat in Eq.2
        # u_hat: [batch_size, 1152, 10, 16]
        u_hat = torch.matmul(W_batch, u_stack).squeeze()

        # Line 2: Initialize b into zeros
        # b: [1152, 10]
        b = Variable(torch.zeros(1152, 10))
        if self.opt.use_cuda & torch.cuda.is_available():
            b = b.cuda()

        # Start Routing
        for r in range(self.opt.r):
            # Line 4: c_i = softmax(b_i)
            # c: [1152, 10]
            c = F.softmax(b, dim=1)
            c = c.unsqueeze(2).unsqueeze(0)
            assert c.size() == torch.Size([1, 1152, 10, 1])

            # Line 5: s_j = sum_i(c_ij * u_hat_j|i)
            # u_hat: [batch_size, 1152, 10, 16]
            # s: [batch_size, 10, 16]
            s = torch.sum(u_hat * c, dim=1)

            # Line 6: v_j = squash(s_j)
            # v: [batch_size, 10, 16]
            v = self.squash(s)
            assert v.size() == torch.Size([batch_size, 10, 16])

            # Line 7: b_ij += u_hat * v_j
            # u_hat: [batch_size, 1152, 10, 16]
            # v: [batch_size, 10, 16]
            # a: [batch_size, 10, 1152, 1]
            a = torch.matmul(u_hat.transpose(1, 2), v.unsqueeze(3))
            # b: [1152, 10]
            b = b + torch.sum(a.squeeze().transpose(1, 2), dim=0)

        return v

    def squash(self, s):
        '''
        Args:
            `s`: [batch_size, 10, 16]

        v_j = (norm(s_j) ^ 2 / (1 + norm(s_j) ^ 2)) * (s_j / norm(s_j))

        Reference: Eq.1 in Section 2.
        '''
        batch_size = s.size(0)

        # s: [batch_size, 10, 16]
        square = s ** 2

        # square_sum for v: [batch_size, 10]
        square_sum = torch.sum(square, dim=2)

        # norm for v: [batch_size, 10]
        norm = torch.sqrt(square_sum)

        # factor for v: [batch_size, 10]
        factor = norm ** 2 / (norm * (1 + norm ** 2))

        # v: [batch_size, 10, 16]
        v = factor.unsqueeze(2) * s
        assert v.size() == torch.Size([batch_size, 10, 16])

        return v
