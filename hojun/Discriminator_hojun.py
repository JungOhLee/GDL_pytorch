import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, input_dim, discriminator_conv_channels, discriminator_conv_kernel_size, 
                 discriminator_conv_strides, discriminator_activation, discriminator_dropout_rate, 
                 discriminator_learning_rate, discriminator_batch_norm_momentum=0.1, discriminator_batch_norm_use=True):
        
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.discriminator_conv_channels=discriminator_conv_channels
        self.discriminator_conv_kernel_size = discriminator_conv_kernel_size
        self.discriminator_conv_strides = discriminator_conv_strides
        self.discriminator_batch_norm_momentum = discriminator_batch_norm_momentum
        self.discriminator_batch_norm_use=discriminator_batch_norm_use
        self.discriminator_activation = discriminator_activation
        self.discriminator_dropout_rate = discriminator_dropout_rate
        self.discriminator_learning_rate = discriminator_learning_rate
        self.n_layers_discriminator = len(discriminator_conv_channels)
        self.discriminator_conv_layers=nn.Sequential()


        # discriminator_input = Input(shape=self.input_dim, name='discriminator_input')

        zero_input=torch.zeros(size=self.input_dim)
        for i in range(self.n_layers_discriminator):
            if i==0:
                self.discriminator_conv_layers.add_module("Conv "+str(i), nn.Conv2d(1, self.discriminator_conv_channels[i], 
                                                                     self.discriminator_conv_kernel_size[i],
                                                                    stride=self.discriminator_conv_strides[i],
                                                                    padding=2))
            else:
                self.discriminator_conv_layers.add_module("Conv "+str(i), nn.Conv2d(self.discriminator_conv_channels[i-1],
                                                                     self.discriminator_conv_channels[i], 
                                                                     self.discriminator_conv_kernel_size[i],
                                                                    stride=self.discriminator_conv_strides[i],
                                                                    padding=2))
            if self.discriminator_batch_norm_use and i > 0:
                self.discriminator_conv_layers.add_module("Batchnorm "+str(i), nn.BatchNorm2d(self.discriminator_conv_channels[i]))                

            self.discriminator_conv_layers.add_module("Activation "+str(i), self.get_activation())
            if self.discriminator_dropout_rate:
                self.discriminator_conv_layers.add_module("Dropout "+str(i), nn.Dropout(p=self.discriminator_dropout_rate))

        self.discriminator_conv_layers.add_module("Flatten", nn.Flatten())
        zero_output=self.discriminator_conv_layers(zero_input)
        output_size=zero_output.size()[1]
        

        self.discriminator_conv_layers.add_module("Fully Connected Layer", nn.Linear(output_size, 1))
    
    
            
    def forward(self, x):
        out=self.discriminator_conv_layers(x)
        return out
             
                
    def get_activation(self, activation='relu'):
        if activation == 'leaky_relu':
            layer = nn.LeakyReLU(negative_slope=0.2)
        else:
            layer = nn.ReLU()
        return layer