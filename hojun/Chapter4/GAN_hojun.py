import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd.variable import Variable
from Discriminator_hojun import Discriminator
from Generator_hojun import *
from utils.loaders import load_safari

import argparse
import numpy as np
import os
import matplotlib.pyplot as plot
import json
import pickle as pkl
import matplotlib.pyplot as plt

def init_params(model):
    for p in model.parameters():
        if(p.dim() > 1):
            nn.init.xavier_normal_(p)
        else:
            nn.init.uniform_(p, 0.1, 0.2)


if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='Basic GAN')
    parser.add_argument('--dropout', type=float, default=0.4, help='dropout rate in training (default: 0.4)')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training (default: 32)')
    parser.add_argument('--max_epochs', type=int, default=10, help='number of max epochs in training (default: 10)')
    parser.add_argument('--lr', type=float, default=4*1e-04, help='learning rate (default: 0.0004)')
    parser.add_argument('--cuda_number', type=int, default=0, help='GPU number (default: 0)')
    
    args = parser.parse_args()
    
    device = torch.device('cuda:'+str(args.cuda_number) if torch.cuda.is_available() else 'cpu')

    
    discriminator = Discriminator(input_dim=(60,1,28,28), discriminator_conv_channels = [64,64,128,128],
        discriminator_conv_kernel_size = [5,5,5,5], discriminator_conv_strides = [2,2,2,1],
        discriminator_batch_norm_momentum = None, discriminator_activation = 'relu',
        discriminator_dropout_rate = args.dropout).to(device)
    generator=Generator(input_dim=100, generator_initial_dense_layer_size = (64, 7, 7)
        , generator_upsample = [2,2, 1, 1]
        , generator_conv_channels = [128,64, 64,1]
        , generator_conv_kernel_size = [5,5,5,5]
        , generator_conv_strides = [1,1, 1, 1]
        , generator_batch_norm_momentum = 0.9
        , generator_activation = 'relu').to(device)
    
    init_params(generator)
    init_params(discriminator)
    
    camel_loader=DataLoader(input_data, batch_size=args.batch_size, shuffle=True)
    # Loss and optimizer
    criterion = nn.BCELoss()

    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr)
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr)

    for epoch in range(args.max_epochs):
        for _, (x_batch, y_batch) in enumerate(camel_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # Forward pass
            if len(x_batch)<args.batch_size:
                valid_discriminator = torch.ones((len(x_batch),1))
                fake_discriminator = torch.zeros((len(x_batch),1))
            else:
                valid_discriminator = torch.ones((args.batch_size,1))
                fake_discriminator = torch.zeros((args.batch_size,1))

            valid_discriminator=Variable(valid_discriminator).to(device)
            fake_discriminator=Variable(fake_discriminator).to(device)
            noise =Variable(torch.normal(0, 1.0, (args.batch_size, 100))).to(device)
            gen_x_batch = generator(noise)
            total_x_batch=torch.cat((gen_x_batch, x_batch), 0)
            predicted_labels=discriminator(total_x_batch)
            labels=torch.cat((fake_discriminator, valid_discriminator), 0)
            predicted_labels=predicted_labels.squeeze(1)
            labels=labels.squeeze(1)

            discriminator_loss = criterion(predicted_labels, labels).to(device)
            # Backward and optimize
            discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            discriminator_optimizer.step()


            valid_generator = Variable(torch.ones([args.batch_size,1])).to(device).squeeze(1)
            noise_generator = Variable(torch.normal(0, 1, [args.batch_size, 100])).to(device)
            output_generator = generator(noise_generator)
            output_generator = discriminator(output_generator).squeeze(1)
            generator_loss=criterion(output_generator, valid_generator)
            generator_optimizer.zero_grad()
            generator_loss.backward()
            generator_optimizer.step()

        print("%d: Discriminator Loss=%.4f, Generator Loss=%.4f" % (epoch, 10000*discriminator_loss.item(), generator_loss.item()))
    