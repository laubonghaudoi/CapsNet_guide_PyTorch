import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from PrimaryCaps import PrimaryCaps
from DigitCaps import DigitCaps
from Decoder import Decoder


class CapsNet(nn.Module):

    def __init__(self, opt):
        '''
        The CapsNet consists of 3 layers: `Conv1`, `PrimaryCaps`, `DigitCaps`.`Conv1`
        is a ordinary 2D convolutional layer with 9x9 kernels, stride 2, 256 output
        channels, and ReLU activations. `PrimaryCaps` and `DigitCaps` are two capsule
        layers with Dynamic Routing between them. For further details of these two
        layers, see the docstrings of their classes. For each [1, 28, 18] input image,
        CapsNet outputs a [16, 10] tensor, representing the 16-dimensional output
        vector from 10 digit capsules.

        Reference: Section 4, Figure 1
        '''
        super(CapsNet, self).__init__()
        self.opt = opt

        self.Conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=9)
        self.PrimaryCaps = PrimaryCaps()
        self.DigitCaps = DigitCaps(opt)

        self.Decoder = Decoder(opt)

    def forward(self, x):
        '''
        Args:
            `x`: [batch_size, 1, 28, 28] A MNIST sample
        
        Return:
            `v`: [batch_size, 10, 16] CapsNet outputs, 16D rediction vectors of
                10 digit capsules

        The dimension transformation procedure of an input tensor in each layer:
            0. Input: [batch_size, 1, 28, 28] -->
            1. `Conv1` --> [batch_size, 256, 20, 20] --> 
            2. `PrimaryCaps` --> [batch_size, 8, 6, 6] x 32 capsules --> 
            3. Flatten, concatenate, squash --> [batch_size, 1152, 8] -->
            4. `W_ij`s and `DigitCaps` --> [batch_size, 16, 10] -->
            5. Length of 10 capsules --> [batch_size, 10] output probabilities
        '''
        # Input: [batch_size, 1, 28, 28]
        x = F.relu(self.Conv1(x))
        # PrimaryCaps input: [batch_size, 256, 20, 20]
        u = self.PrimaryCaps(x)
        # PrimaryCaps output u: [batch_size, 1152, 8]
        v = self.DigitCaps(u)
        # DigitCaps output v: [batsh_size, 10, 16]
        return v

    def marginal_loss(self, v, target, l=0.5):
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
        marginal_loss = torch.sum(marginal_loss)

        return marginal_loss

    def reconstruction_loss(self, reconstruction, image):
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
        assert image.size() == (batch_size, 784)
        
        # Scalar Variable
        reconstruction_loss = torch.sum((reconstruction - image) ** 2)
        return reconstruction_loss

    def loss(self, v, target, image):
        '''
        Args:
            `v`: [batch_size, 10, 16] CapsNet outputs
            `target`: [batch_size, 10] One-hot MNIST labels
            `image`: [batch_size, 1, 28, 28] MNIST samples

        Return:
            `L`: Scalar Variable, total loss
            `marginal_loss`: Scalar Variable
            `reconstruction_loss`: Scalar Variable

        The reconstruction loss is scaled down by 5e-4, serving as
        a regularization method.

        Reference: Section 4.1
        '''
        batch_size = image.size(0)

        marginal_loss = self.marginal_loss(v, target)

        # Get reconstructions from the decoder network
        reconstruction = self.Decoder(v, target)
        reconstruction_loss = self.reconstruction_loss(reconstruction, image)

        # Scalar Variable
        loss = (marginal_loss + 0.0005 * reconstruction_loss) / batch_size

        return loss, marginal_loss / batch_size, reconstruction_loss / batch_size
