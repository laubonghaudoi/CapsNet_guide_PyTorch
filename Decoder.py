import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    '''
    The decoder network consists of 3 fully connected layers. For each
    [10, 16] output, we mask out the incorrect predictions, and send
    the [16,] vector to the decoder network to reconstruct a [784,] size
    image.

    Reference: Section 4.1, Fig. 2
    '''

    def __init__(self, opt):
        '''
        The decoder network consists of 3 fully connected layers, with
        512, 1024, 784 neurons each.
        '''
        super(Decoder, self).__init__()
        self.opt = opt
        self.pose_len = opt.pose**2

        self.fc1 = nn.Linear(self.pose_len, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, self.opt.img_size**2)

    def forward(self, v, target):
        '''
        Args:
            v: [batch_size, 10, 16]
            target: [batch_size, 10]

        Return:
            `reconstruction`: [batch_size, 784]

        We send the outputs of the `DigitCaps` layer, which is a
        [batch_size, 10, 16] size tensor into the decoder network, and
        reconstruct a [batch_size, 784] size tensor representing the image.
        '''
        batch_size = target.size(0)

        target = target.type(torch.FloatTensor)
        # mask: [batch_size, 10, 16]
        mask = torch.stack([target for i in range(self.pose_len)], dim=2)
        assert mask.size() == torch.Size([batch_size, self.opt.num_class, self.pose_len])
        if self.opt.use_cuda & torch.cuda.is_available():
            mask = mask.cuda()

        # v: [bath_size, 10, 16]
        v_masked = mask * v
        v_masked = torch.sum(v_masked, dim=1)
        assert v_masked.size() == torch.Size([batch_size, self.pose_len])

        # Forward
        v = F.relu(self.fc1(v_masked))
        v = F.relu(self.fc2(v))
        reconstruction = torch.sigmoid(self.fc3(v))

        assert reconstruction.size() == torch.Size([batch_size, self.opt.img_size**2])
        return reconstruction
