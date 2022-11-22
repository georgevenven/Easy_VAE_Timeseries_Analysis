import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile
from PIL import Image
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import io
import librosa

class Encoder(nn.Module):
    def __init__(self, latent_dims):
        super(Encoder, self).__init__()
        self.batchNorm1 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 8, 3, 1, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 8, 3, 2, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8,16,3, 1, padding=1)
        self.batchNorm4 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16,16,3,2, padding=1)
        self.batchNorm5 = nn.BatchNorm2d(16)
        self.conv5 = nn.Conv2d(16,24,3,1, padding=1)
        self.batchNorm6 = nn.BatchNorm2d(24)
        self.conv6 = nn.Conv2d(24,24,3,2, padding=1)
        self.batchNorm7 = nn.BatchNorm2d(24)
        self.conv7 = nn.Conv2d(24,32,3,1, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8192, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 64)
        
        self.fc_mu = nn.Linear(in_features=64, out_features=latent_dims)
        self.fc_logvar = nn.Linear(in_features=64, out_features=latent_dims)
            
    def forward(self, x):
        x = F.relu(self.conv1(self.batchNorm1(x)))
        x = F.relu(self.conv2(self.batchNorm2(x)))
        x = F.relu(self.conv3(self.batchNorm3(x)))
        x = F.relu(self.conv4(self.batchNorm4(x)))
        x = F.relu(self.conv5(self.batchNorm5(x)))
        x = F.relu(self.conv6(self.batchNorm6(x)))
        x = F.relu(self.conv7(self.batchNorm7(x)))

        x = self.flatten(x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)

        return x_mu, x_logvar

class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(in_features=latent_dims, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=1024)
        self.fc4 = nn.Linear(in_features=1024, out_features=8192)
        self.unflatten = nn.Unflatten(1, (32, 16, 16))

        self.conv1 = nn.ConvTranspose2d(32,24,3,1,padding=1)
        self.conv2 = nn.ConvTranspose2d(24,24,3,2,padding=1,output_padding=1)
        self.conv3 = nn.ConvTranspose2d(24,16,3,1,padding=1)
        self.conv4 = nn.ConvTranspose2d(16,16,3,2,padding=1,output_padding=1)
        self.conv5 = nn.ConvTranspose2d(16,8,3,1,padding=1)
        self.conv6 = nn.ConvTranspose2d(8,8,3,2,padding=1,output_padding=1)
        self.conv7 = nn.ConvTranspose2d(8,1,3,1,padding=1)
        self.batchNorm1 = nn.BatchNorm2d(32)
        self.batchNorm2 = nn.BatchNorm2d(24)
        self.batchNorm3 = nn.BatchNorm2d(24)
        self.batchNorm4 = nn.BatchNorm2d(16)
        self.batchNorm5 = nn.BatchNorm2d(16)
        self.batchNorm6 = nn.BatchNorm2d(8)
        self.batchNorm7 = nn.BatchNorm2d(8)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        
        x = self.unflatten(x)

        x = F.relu(self.conv1(self.batchNorm1(x)))
        x = F.relu(self.conv2(self.batchNorm2(x)))
        x = F.relu(self.conv3(self.batchNorm3(x)))
        x = F.relu(self.conv4(self.batchNorm4(x)))
        x = F.relu(self.conv5(self.batchNorm5(x)))
        x = F.relu(self.conv6(self.batchNorm6(x)))
        x = F.relu(self.conv7(self.batchNorm7(x)))

        x = F.relu(x) # last layer before output is sigmoid, since we are using BCE as reconstruction loss

        return x
    
class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder(latent_dims)
        self.decoder = Decoder(latent_dims)
    
    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)

        latent = self.latent_sample(latent_mu, latent_logvar)

        x_recon = self.decoder(latent)

        return x_recon, latent_mu, latent_logvar
    
    def latent_sample(self, mu, logvar):
        if self.training:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu
    
def vae_loss(recon_x, x, mu, logvar):
    flatten = nn.Flatten()
    flattened_recon = flatten(recon_x)
    flattened_x = flatten(x)
    recon_loss = F.mse_loss(flattened_recon, flattened_x, reduction='sum')
    kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + kldivergence * 1

class VAE():
    # takes the parameters from the front end to initalize the vae
    # inits the vae
    # gets gpus 
    def __init__(self, dir, input_type, remove_empty_noise, empty_threshold, empty_window_size, segment_length, segment_overlap, segment_size, latent_dims, batch_size, epochs, train_valid_split, save_epoch_interval, root_dir):
        vae = VariationalAutoencoder(latent_dims=latent_dims)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        vae = vae.to(device)        
        
        self.dir = dir
        self.root_dir = root_dir
        self.input_type = input_type
        self.remove_empty_noise = remove_empty_noise
        self.empty_threshold = empty_threshold
        self.empty_window_size = empty_window_size
        self.segment_length = segment_length
        self.segment_overlap = segment_overlap
        self.segment_size = segment_size
        self.latent_dims = latent_dims
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_valid_split = train_valid_split
        self.save_epoch_interval = save_epoch_interval
        self.vae = vae

        self.load_data()
        self.train()

    # loads data from the directory specified, 
    def load_data():
        # load data from the data_dir
        pass

    # trains the model 
    def train():
        pass 
