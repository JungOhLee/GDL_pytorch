import torch
import torch.nn as nn
import numpy as np

class Generator(nn.Module):
    def __init__(self, input_dim, generator_initial_dense_layer_size,
                generator_upsample,
                generator_conv_channels, 
                generator_conv_kernel_size, 
                generator_conv_strides, 
                generator_batch_norm_momentum, 
                generator_activation,  
                generator_batch_norm_use=True):
        
        super(Generator, self).__init__()
        self.input_dim=input_dim
        self.generator_initial_dense_layer_size = generator_initial_dense_layer_size
        self.generator_upsample = generator_upsample
        self.generator_conv_channels = generator_conv_channels
        self.generator_conv_kernel_size = generator_conv_kernel_size
        self.generator_conv_strides = generator_conv_strides
        self.generator_batch_norm_momentum = generator_batch_norm_momentum
        self.generator_activation = generator_activation
        self.n_layers_generator = len(generator_conv_channels)
        self.get_activation=get_activation
        
        self.initial_output_size=np.prod(self.generator_initial_dense_layer_size)
        self.generator_layer1=nn.Sequential(
            nn.Linear(input_dim, self.initial_output_size),
            nn.BatchNorm1d(self.initial_output_size),
            self.get_activation()
        )
        self.generator_layer2=make_layers(in_channel=self.generator_initial_dense_layer_size[0],
                                         out_channel=self.generator_conv_channels[0],
                                         kernel_size=self.generator_conv_kernel_size[0],
                                         padding=2)
        self.generator_layer3=make_layers(in_channel=self.generator_conv_channels[0],
                                         out_channel=self.generator_conv_channels[1],
                                         kernel_size=self.generator_conv_kernel_size[1],
                                         padding=2)
        self.generator_layer4=nn.Sequential(
            nn.Conv2d(in_channels=self.generator_conv_channels[1], out_channels=self.generator_conv_channels[2],
                     kernel_size=self.generator_conv_kernel_size[2], padding=2),
            nn.BatchNorm2d(self.generator_conv_channels[2]),
            self.get_activation()
        )
        self.generator_layer5=nn.Sequential(
            nn.Conv2d(in_channels=self.generator_conv_channels[2], out_channels=self.generator_conv_channels[3],
                     kernel_size=self.generator_conv_kernel_size[3], padding=2),
            nn.BatchNorm2d(self.generator_conv_channels[3])
        )
        
    def forward(self, x):
        x=self.generator_layer1(x)
        x.view(self.generator_initial_dense_layer_size)
        x=self.generator_layer2(x)
        x=self.generator_layer3(x)
        x=self.generator_layer4(x)
        x=self.generator_layer5(x)
        return x


        

def get_activation(self, activation='relu'):
    if activation == 'leaky_relu':
        layer = nn.LeakyReLU(negative_slope=0.2)
    else:
        layer = nn.ReLU()
    return layer
        
def make_layers(in_channel, out_channel, kernel_size, padding):
    return_layer=nn.Sequential(
        nn.Upsample(scale_factor=2),
        nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm2d(out_channel),
        get_activation()
    )
    
    return return_layer