import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from PrimaryCaps import PrimaryCaps
from ConvCaps import ConvCaps
from ClassCaps import ClassCaps
from Decoder import Decoder
from capsule_utils import marginal_loss, spread_loss, reconstruction_loss

class CapsNet(nn.Module):

    def __init__(self, opt):
        super(CapsNet, self).__init__()
        self.opt = opt

        self.Conv1 = nn.Conv2d(in_channels=opt.in_channel, out_channels=opt.A,
                               kernel_size=opt.conv1_ksize, stride=opt.conv1_stride)
        self.PrimaryCaps = PrimaryCaps(opt, opt.A, opt.B)
        self.ConvCaps1 = ConvCaps(opt, opt.B, opt.C, kernel=opt.convcaps1_ksize, stride=opt.convcaps1_stride)
        self.ConvCaps2 = ConvCaps(opt, opt.C, opt.D, kernel=opt.convcaps2_ksize, stride=opt.convcaps2_stride)
        self.ClassCaps = ClassCaps(opt, opt.D, opt.num_class)

        self.Decoder = Decoder(opt)

    def forward(self, x):
        x = F.relu(self.Conv1(x))
        capsules = self.PrimaryCaps.forward(x)
        capsules = self.ConvCaps1.forward(capsules)
        capsules = self.ConvCaps2.forward(capsules)
        capsules = self.ClassCaps.forward(capsules)
        return capsules

    def loss(self, capsules, targets, data):
        batch_size = self.opt.batch_size
        main_loss = marginal_loss(capsules, targets, self.opt) if self.opt.loss == 'marginal_loss' \
                                                           else spread_loss(capsules[:,:,-1], targets, self.opt.margin)
        # Get reconstructions from the decoder network
        reconstruction = self.Decoder(capsules[:,:,:-1], targets)
        recon_loss = reconstruction_loss(reconstruction, data, self.opt)

        # Scalar Variable
        loss = (main_loss + self.opt.alpha * recon_loss) / batch_size
        return loss, main_loss / batch_size, recon_loss / batch_size
