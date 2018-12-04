"""
A tutorial-style implementation of CapsNet in PyTorch.

Paper link: https://arxiv.org/abs/1710.09829v2
Paper link: https://openreview.net/pdf?id=HJWLfGWRb

@author laubonghaudoi
@author Alexbanana19

For better understanding, read the codes and comments in the following order:

1. `__main__` in `main.py`
2. `utils.py`
3. `capsule_utils.py`
3. `CapsNet.__init__()` and `CapsNet.forward()` in `CapsNet.py`
4. `PrimaryCaps.py`
5. `ConvCaps.py`
6. `ClassCaps.py`
7. `Decoder.py`
9. `train()` and `test()` in `main.py`

You might find helpful with the paper *Dynamic Routing Between Capsules* and
*Matrix Capsules with EM Routing* at your hand for referencing.
"""

import os
import time
from tqdm import *

import torch
import torch_extras
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from torch.autograd import Variable

from CapsNet import CapsNet
from utils import get_opts, get_dataloader

# PyTorch does not provide one-hot vector conversion, we achieve this
# by pytorch-extras
setattr(torch, 'one_hot', torch_extras.one_hot)

'''TODO: lambda and margin schedule'''

def train(opt, train_loader, test_loader, model, writer):
    num_data = len(train_loader.dataset)
    num_batches = len(train_loader)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)

    model.train()
    for epoch in range(opt.epochs):
        # Update learning rate
        scheduler.step()
        print('Learning rate: {}'.format(scheduler.get_lr()[0]))

        start_time = time.time()
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            batch_size = data.size(0)
            global_step = batch_idx + epoch * num_batches

            # Transform to one-hot indices: [batch_size, 10]
            target = torch.one_hot((batch_size, opt.num_class), target.view(-1, 1))
            assert target.size() == torch.Size([batch_size, opt.num_class])

            # Use GPU if available
            data, target = Variable(data), Variable(target)
            if opt.use_cuda & torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            # Train step
            optimizer.zero_grad()
            output = model(data)

            L, m_loss, r_loss = model.loss(output, target, data)
            L.backward()

            optimizer.step()

            # Log losses
            writer.add_scalar('train/loss', L.item(), global_step)
            writer.add_scalar('train/main_loss', m_loss.item(), global_step)
            writer.add_scalar('train/reconstruction_loss', r_loss.item(), global_step)

            # Print losses
            if batch_idx % opt.print_every == 0:
                tqdm.write('Epoch: {}    Loss: {:.6f}   Main loss: {:.6f}   Recons. loss: {:.6f}'.format(
                    epoch, L.item(), m_loss.item(), r_loss.item()))

        # Print time elapsed for every epoch
        end_time = time.time()
        print('Epoch {} takes {:.0f} seconds.'.format(
            epoch, end_time - start_time))

        # Test model
        test(opt, test_loader, model, writer, epoch, num_batches)

def test(opt, test_loader, model, writer, epoch, num_batches):
    loss = 0
    margin_loss = 0
    recons_loss = 0

    correct = 0

    step = epoch * num_batches + num_batches
    model.eval()
    for data, target in test_loader:
        # Store the indices for calculating accuracy
        label = target.unsqueeze(0).type(torch.LongTensor)

        batch_size = data.size(0)
        # Transform to one-hot indices: [batch_size, 10]
        target = torch.one_hot((batch_size, opt.num_class), target.view(-1, 1))
        assert target.size() == torch.Size([batch_size, opt.num_class])

        # Use GPU if available
        data, target = Variable(data, volatile=True), Variable(target)
        if opt.use_cuda & torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        # Output predictions
        output = model(data)
        L, m_loss, r_loss = model.loss(output, target, data)
        loss += L
        margin_loss += m_loss
        recons_loss += r_loss

        # Count correct numbers
        # norms: [batch_size, 10, 16]
        norms = torch.sqrt(torch.sum(output**2, dim=2))
        # pred: [batch_size,]
        pred = norms.data.max(1, keepdim=True)[1].type(torch.LongTensor)
        correct += pred.eq(label.view_as(pred)).cpu().sum()

    # Visualize reconstructed images of the last batch
    recons = model.Decoder(output, target)
    recons = recons.view(batch_size, 1, 28, 28)
    recons = vutils.make_grid(recons.data, normalize=True, scale_each=True)
    writer.add_image('Image-{}'.format(step), recons, step)

    # Log test losses
    loss /= len(test_loader)
    margin_loss /= len(test_loader)
    recons_loss /= len(test_loader)
    acc = correct / len(test_loader.dataset)
    writer.add_scalar('test/loss', loss.item(), step)
    writer.add_scalar('test/main_loss', margin_loss.item(), step)
    writer.add_scalar('test/reconstruction_loss', recons_loss.item(), step)
    writer.add_scalar('test/accuracy', acc, step)

    # Print test losses
    print('\nTest loss: {:.4f}   Marginal loss: {:.4f}   Recons loss: {:.4f}'.format(
        loss.item(), margin_loss.item(), recons_loss.item()))
    print('Accuracy: {}/{} ({:.0f}%)\n'.format(correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    # Checkpoint model
    torch.save(model, './ckpt/epoch_{}-loss_{:.6f}-acc_{:.6f}.pt'.format(
        epoch, loss.item(), acc))


if __name__ == "__main__":
    # Default configurations
    opt = get_opts()
    train_loader, test_loader = get_dataloader(opt)

    # Initialize CapsNet
    model = CapsNet(opt)

    # Enable GPU usage
    if opt.use_cuda & torch.cuda.is_available():
        model.cuda()

    # Print the model architecture and parameters
    print("Model architectures: ")
    print(model)

    print("\nSizes of parameters: ")
    for name, param in model.named_parameters():
        print("{}: {}".format(name, list(param.size())))
    n_params = sum([p.nelement() for p in model.parameters()])
    # The coupling coefficients b_ij are not included in the parameter list,
    # we need to add them mannually, which is 1152 * 10 = 11520.
    print('\nTotal number of parameters: %d \n' % (n_params+11520))

    # Make model checkpoint directory
    if not os.path.exists('ckpt'):
        os.makedirs('ckpt')

    # Start training
    writer = SummaryWriter()
    train(opt, train_loader, test_loader, model, writer)
    writer.close()
