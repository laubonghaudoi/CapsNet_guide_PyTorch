import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

''' Routing Methods for Capsules'''

def kmeans_routing(opt, v):
    ''' The routing method in `Dynamic Routing Between Capsules` '''
    b = Variable(torch.zeros(1152, 10))
    if opt.use_cuda & torch.cuda.is_available():
        b = b.cuda()
    for i in range(self.opt.iter):
        c = F.softmax(b, dim=1)
        c = c.unsqueeze(2).unsqueeze(0)
        s = torch.sum(u_hat * c, dim=1)
        v = self.squash(s)
        a = torch.matmul(u_hat.transpose(1, 2), v.unsqueeze(3))
        b = b + torch.sum(a.squeeze().transpose(1, 2), dim=0)
    return v

def em_routing(V, a, beta_u, beta_a, opt):
    batch_size = V.size(0)
    N1 = V.size(1) # Number of votes in layer L
    N2 = V.size(2) # Number of clustering centers in layer L+1
    r = 1.*torch.ones([batch_size, N1, N2, 1])/N2
    if opt.use_cuda & torch.cuda.is_available():
        r.cuda()

    for i in range(opt.iter):
        # M-step
        r = r * a
        r = r.clamp(0.01) # prevent nan
        r_sum = r.sum(1, True)
        mu = torch.sum(r * V, 1, True)/r_sum
        sigma = torch.sum(r * (V - mu)**2, 1, True)/r_sum
        sigma = sigma.clamp(0.01) # prevent nan
        cost = (beta_u + torch.log(sigma)) * r_sum
        a = torch.sigmoid(opt.lambda_*(beta_a-torch.sum(cost, 3, True)))

        # E-step
        p = torch.exp(-(V-mu)**2/(2*sigma**2))/torch.sqrt(2*3.14*sigma**2)
        p = p.prod(-1, True)
        ap = a * p
        ap_sum = ap.sum(2, True) + 1e-8
        r = ap / ap_sum
    return mu, a

''' Activation Functions for Capsules '''

def squash(u):
    batch_size = u.size(0)
    square = u ** 2
    square_sum = torch.sum(square, dim=2)
    norm = torch.sqrt(square_sum)
    factor = norm ** 2 / (norm * (1 + norm ** 2))
    u_squashed = factor.unsqueeze(2) * u
    return u_squashed

def sigmoid(cost):
    return torch.sigmoid(cost)


''' Loss Functions '''

def spread_loss(v, target, m=0.2):
    b = v.size(0)
    tar_ind = target.argmax(1).view(b)
    a_t = v[list(range(b)),tar_ind].view(b, 1)
    loss = torch.mean(torch.sum(torch.max(torch.zeros_like(v), m-(a_t-v))**2, 1))
    return loss

def marginal_loss(v, target, l=0.5):
    '''
    Args:
        `v`: [batch_size, 10, 16]
        `target`: [batch_size, 10]
        `l`: Scalar, lambda for down-weighing the loss for absent digit classes

    Return:
        `marginal_loss`: Scalar

    L_c = T_c * max(0, m_plus - norm(v_c)) ^ 2 + lambda * (1 - T_c) * max(0, norm(v_c) - m_minus) ^2

    Reference: Eq.4 in Section 3.
    '''
    batch_size = v.size(0)

    square = v ** 2
    square_sum = torch.sum(square, dim=2)
    # norm: [batch_size, 10]
    norm = torch.sqrt(square_sum)
    assert norm.size() == torch.Size([batch_size, 10])

    # The two T_c in Eq.4
    T_c = target.type(torch.FloatTensor)
    zeros = Variable(torch.zeros(norm.size()))
    # Use GPU if available
    if self.opt.use_cuda & torch.cuda.is_available():
        zeros = zeros.cuda()
        T_c = T_c.cuda()

    # Eq.4
    marginal_loss = T_c * (torch.max(zeros, 0.9 - norm) ** 2) + \
        (1 - T_c) * l * (torch.max(zeros, norm - 0.1) ** 2)
    marginal_loss = torch.mean(torch.sum(marginal_loss, 1))

    return marginal_loss

def reconstruction_loss(reconstruction, image, opt):
    '''
    Args:
        `reconstruction`: [batch_size, 784] Decoder outputs of images
        `image`: [batch_size, 1, 28, 28] MNIST samples

    Return:
        `reconstruction_loss`: Scalar Variable

    The reconstruction loss is measured by a squared differences
    between the reconstruction and the original image.

    Reference: Section 4.1
    '''
    batch_size = image.size(0)
    # image: [batch_size, 784]
    image = image.view(batch_size, -1)
    assert image.size() == (batch_size, opt.img_size**2)

    # Scalar Variable
    recon_loss = torch.mean(torch.sum((reconstruction - image) ** 2, 1))
    return recon_loss
