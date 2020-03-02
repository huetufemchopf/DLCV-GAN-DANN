import torch, os
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dset
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.utils import make_grid
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import parser1
import numpy as np
import data_gan
import model_gan
#%matplotlib inline

lr = 0.0002
max_epoch = 8
batch_size = 32
z_dim = 100
image_size = 64
g_conv_dim = 64
d_conv_dim = 64
log_step = 100
sample_step = 500
sample_num = 32
SAMPLE_PATH = '/home/celine/Documents/DLCV/hw3-huetufemchopf/hw3_data/face/train/sample'
def save_model(model, save_path):
    torch.save(model.state_dict(),save_path)
    # torch.save(model.module.state_dict(), save_path) # 2 GPUs



if __name__ == '__main__':

    args = parser1.arg_parse()

    '''create directory to save trained model and other info'''
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)

    ''' setup GPU '''
    torch.cuda.set_device(args.gpu)

    ''' setup random seed '''
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    ''' load dataset and prepare data loader '''
    print('===> prepare dataloader ...')
    train_loader = torch.utils.data.DataLoader(data_gan.DataLoaderSegmentation(args, mode='train'),
                                               batch_size=args.train_batch,
                                               num_workers=args.workers,
                                               shuffle=False)

    D = model_gan.Discriminator()
    D.cuda()

    G = model_gan.Generator(z_dim, g_conv_dim)
    G.cuda()

    criterion = nn.BCELoss().cuda()
    d_optimizer = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    g_optimizer = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))


    # denormalization : [-1,1] -> [0,1]
    # normalization : [0,1] -> [-1,1]
    def denorm(x):
        out = (x + 1) / 2
        return out.clamp(0, 1)


#
# try:
#     G.load_state_dict(torch.load('generator.pkl'))
#     D.load_state_dict(torch.load('discriminator.pkl'))
#     print("\n-------------model restored-------------\n")
# except:
#     print("\n-------------model not restored-------------\n")
#     pass

total_batch = len(train_loader.dataset) // batch_size
fixed_z = Variable(torch.randn(sample_num, z_dim)).cuda()
for epoch in range(max_epoch):
    for i, (imgs, filename) in enumerate(train_loader):
        # Build mini-batch dataset
        imgs = imgs.cuda()
        # Create the labels which are later used as input for the BCE loss
        real_labels = Variable(torch.ones(args.train_batch)).cuda()
        fake_labels = Variable(torch.zeros(args.train_batch)).cuda()

        # ============ train the discriminator ============
        # Compute BCE_loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
        # Second term of the loss is always zero since real_labels = 1
        outputs = D(imgs)
        d_loss_real = criterion(outputs, real_labels)  # BCE
        real_score = outputs

        # compute BCE_loss using fake images
        z = Variable(torch.randn(batch_size, z_dim)).cuda()
        fake_images = G(z)
        outputs2 = D(fake_images)
        d_loss_fake = criterion(outputs2, fake_labels)  # BCE
        fake_score = outputs2

        # Backprop + Optimize
        d_loss = d_loss_real + d_loss_fake
        D.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # ============ train the generator ============
        # Compute loss with fake images
        z = Variable(torch.randn(batch_size, z_dim)).cuda()
        fake_images = G(z)
        outputs3 = D(fake_images)

        # We train G to maximize log(D(G(z))) instead of minimizing log(1-D(G(z)))
        # For the reason, see the last paragraph of section 3. https://arxiv.org/pdf/1406.2661.pdf
        g_loss = criterion(outputs3, real_labels)  # BCE

        # Backprob + Optimize
        D.zero_grad()
        G.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        # if (i + 1) % log_step == 0:
        #     print("Epoch [%d/%d], Step[%d/%d], d_loss: %.4f, g_loss: %.4f, D(x): %.2f, D(G(z)): %.2f" % (
        #        epoch, max_epoch, i + 1, total_batch, d_loss.data[0].item(), g_loss.data[0].item(), real_score.data.mean().item(),
        #        fake_score.data.mean().item()))

        if (i + 1) % sample_step == 0:
            real_images = imgs
            torchvision.utils.save_image(denorm(imgs.data),
                                         os.path.join(args.sample_dir, 'real_samples-%d-%d.png') % (
                                             epoch + 1, i + 1), nrow=8)

        if (i + 1) % sample_step == 0:
            fake_images = G(fixed_z)
            torchvision.utils.save_image(denorm(fake_images.data),
                                         os.path.join(args.sample_dir, 'fake_samples-%d-%d.png') % (
                                             epoch + 1, i + 1), nrow=8)

save_model(G, os.path.join(args.save_dir, 'G_{}.pth.tar'.format(epoch)))
save_model(D, os.path.join(args.save_dir, 'G_{}.pth.tar'.format(epoch)))

#torch.save(G.state_dict(), 'generator.pkl')
# torch.save(D.state_dict(), 'discriminator.pkl')