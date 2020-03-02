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
import data_acgan
import model_acgan


if __name__ == '__main__':

    args = parser1.arg_parse()

    '''create directory to save trained model and other info'''
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.acgan_dir):
        os.makedirs(args.acgan_dir)
        ''' setup GPU '''

    torch.cuda.set_device(args.gpu)

    ''' setup random seed '''
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    ''' load dataset and prepare data loader '''
    print('===> prepare dataloader ...')
    train_loader = torch.utils.data.DataLoader(data_acgan.DataLoaderACGAN(args, mode='train'),
                                               batch_size=args.train_batch,
                                               num_workers=args.workers,
                                               shuffle=False)

    adversarial_loss = torch.nn.BCELoss()
    auxiliary_loss = torch.nn.BCELoss()

    # Initialize generator and discriminator
    generator = model_acgan.Generator()
    discriminator = model_acgan.Discriminator()


    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    auxiliary_loss.cuda()

    # Initialize weights
    generator.apply(model_acgan.weights_init_normal)
    discriminator.apply(model_acgan.weights_init_normal)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor


    def sample_image(n_row, batches_done):
        """Saves a grid of generated digits ranging from 0 to n_classes"""
        # Sample noise
        z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, 100))))
        # Get labels ranging from 0 to n_classes for n rows
        labels = np.array([num for _ in range(n_row) for num in range(n_row)])
        labels = Variable(LongTensor(labels))
        gen_imgs = generator(z.cuda(), labels.cuda())
        print(gen_imgs)
        exit()
        save_image(gen_imgs.data,  "images/%d.png" % batches_done, nrow=n_row, normalize=True)

for epoch in range(args.epoch):
    for i, (imgs, filename, labels) in enumerate(train_loader):

        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, 100))))
        gen_labels = Variable(LongTensor(np.random.randint(0,2, batch_size)))

        # Generate a batch of images
        gen_imgs = generator(z, gen_labels)


        # Loss measures generator's ability to fool the discriminator

        validity, pred_label = discriminator(gen_imgs.cuda())
        loss1 = 0.5 * adversarial_loss(validity, valid)
        loss2 = auxiliary_loss(pred_label.type(torch.cuda.FloatTensor), gen_labels.type(torch.cuda.FloatTensor))
        g_loss = 0.5 * (loss1 + loss2)
        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        real_pred, real_aux = discriminator(real_imgs)

        d_real_loss = (adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux.type(torch.cuda.FloatTensor), labels.type(torch.cuda.FloatTensor))) / 2

        # Loss for fake images
        fake_pred, fake_aux = discriminator(gen_imgs.detach())
        d_fake_loss = (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux.type(torch.cuda.FloatTensor), gen_labels.type(torch.cuda.FloatTensor))) / 2

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        # Calculate discriminator accuracy
        print(labels.reshape(-1), gen_labels)

        pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
        gt = np.concatenate([labels.reshape(-1).data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
        d_acc = np.mean(np.argmax(pred, axis=1) == gt)
        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %d%%] [G loss: %f]"
            % (epoch, args.epoch, i, len(train_loader), d_loss.item(), 100 * d_acc, g_loss.item())
        )
        batches_done = epoch * len(train_loader) + i
        z_fixed = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, 100))))
        gen_labels2 = Variable(LongTensor(np.random.randint(0, 2, batch_size)))
        if batches_done % 400 == 0:
            n_row = 2
            imgs = generator(z_fixed, gen_labels2)
            torchvision.utils.save_image(imgs.data,
                             os.path.join(args.acgan_dir, 'fake_samples-%d-%d.png') % (
                                 epoch + 1, i + 1), nrow=8)
            # sample_image(n_row=10, batches_done=batches_done)

