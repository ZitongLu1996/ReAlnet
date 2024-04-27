#!/usr/bin/env python
# coding: utf-8

import math
from collections import OrderedDict
import torch
from torch import nn
from torchvision import transforms
from torchmetrics.functional import pearson_corrcoef, spearman_corrcoef
import torch.utils.model_zoo
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch.nn.functional as F
import time
from tqdm.auto import tqdm
import h5py
import random
import clip
from scipy.stats import spearmanr
from torchmetrics.functional.regression import spearman_corrcoef

device = 'cuda'


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


class Flatten(nn.Module):

    """
    Helper module for flattening input tensor to 1-D for the use in Linear modules
    """

    def forward(self, x):
        return x.view(x.size(0), -1)


class Identity(nn.Module):

    """
    Helper module that stores the current tensor. Useful for accessing by name
    """

    def forward(self, x):
        return x


class CORblock_S(nn.Module):

    scale = 4  # scale of the bottleneck convolution channels

    def __init__(self, in_channels, out_channels, times=1):
        super().__init__()

        self.times = times

        self.conv_input = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.skip = nn.Conv2d(out_channels, out_channels,
                              kernel_size=1, stride=2, bias=False)
        self.norm_skip = nn.BatchNorm2d(out_channels)

        self.conv1 = nn.Conv2d(out_channels, out_channels * self.scale,
                               kernel_size=1, bias=False)
        self.nonlin1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels * self.scale, out_channels * self.scale,
                               kernel_size=3, stride=2, padding=1, bias=False)
        self.nonlin2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(out_channels * self.scale, out_channels,
                               kernel_size=1, bias=False)
        self.nonlin3 = nn.ReLU(inplace=True)

        self.output = Identity()  # for an easy access to this block's output

        # need BatchNorm for each time step for training to work well
        for t in range(self.times):
            setattr(self, f'norm1_{t}', nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f'norm2_{t}', nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f'norm3_{t}', nn.BatchNorm2d(out_channels))

    def forward(self, inp):
        x = self.conv_input(inp)

        for t in range(self.times):
            if t == 0:
                skip = self.norm_skip(self.skip(x))
                self.conv2.stride = (2, 2)
            else:
                skip = x
                self.conv2.stride = (1, 1)

            x = self.conv1(x)
            x = getattr(self, f'norm1_{t}')(x)
            x = self.nonlin1(x)

            x = self.conv2(x)
            x = getattr(self, f'norm2_{t}')(x)
            x = self.nonlin2(x)

            x = self.conv3(x)
            x = getattr(self, f'norm3_{t}')(x)

            x += skip
            x = self.nonlin3(x)
            output = self.output(x)

        return output


def CORnet_S():
    model = nn.Sequential(OrderedDict([
        ('V1', nn.Sequential(OrderedDict([  # this one is custom to save GPU memory
            ('conv1', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                            bias=False)),
            ('norm1', nn.BatchNorm2d(64)),
            ('nonlin1', nn.ReLU(inplace=True)),
            ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ('conv2', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
                            bias=False)),
            ('norm2', nn.BatchNorm2d(64)),
            ('nonlin2', nn.ReLU(inplace=True)),
            ('output', Identity())
        ]))),
        ('V2', CORblock_S(64, 128, times=2)),
        ('V4', CORblock_S(128, 256, times=4)),
        ('IT', CORblock_S(256, 512, times=2)),
        ('decoder', nn.Sequential(OrderedDict([
            ('avgpool', nn.AdaptiveAvgPool2d(1)),
            ('flatten', Flatten()),
            ('linear', nn.Linear(512, 1000)),
            ('output', Identity())
        ])))
    ]))

    # weight initialization
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        # nn.Linear is missing here because I originally forgot
        # to add it during the training of this network
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return model


class Encoder(nn.Module):
    def __init__(self, realnet, n_output):
        super(Encoder, self).__init__()
        
        # CORnet
        self.realnet = realnet
        
        # full connected layer
        self.fc_v1 = nn.Linear(200704, 128)
        self.fc_v2 = nn.Linear(100352, 128)
        self.fc_v4 = nn.Linear(50176, 128)
        self.fc_it = nn.Linear(25088, 128)
        self.fc = nn.Linear(512, n_output)
        self.activation = nn.ReLU()
        
    def forward(self, imgs):
        
        outputs = self.realnet(imgs)
        
        N = len(imgs)
        v1_outputs = self.realnet.module.V1(imgs) # N * 64 * 56 * 56
        v2_outputs = self.realnet.module.V2(v1_outputs) # N * 128 * 28 * 28
        v4_outputs = self.realnet.module.V4(v2_outputs) # N * 256 * 14 * 14
        it_outputs = self.realnet.module.IT(v4_outputs) # N * 512 * 7 * 7
        v1_features = self.fc_v1(v1_outputs.view(N, -1))
        v1_features = self.activation(v1_features)
        v2_features = self.fc_v2(v2_outputs.view(N, -1))
        v2_features = self.activation(v2_features)
        v4_features = self.fc_v4(v4_outputs.view(N, -1))
        v4_features = self.activation(v4_features)
        it_features = self.fc_it(it_outputs.view(N, -1))
        it_features = self.activation(it_features)
        features = torch.cat((v1_features, v2_features, v4_features, it_features), dim=1)
        features = self.fc(features)
        
        return outputs, features


torch.set_default_dtype(torch.float32)

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
)

# this cornet will be used for getting imagenet-based outputs as the classification targets
cornet = CORnet_S().to(device)
cornet = torch.nn.DataParallel(cornet)
url = f'https://s3.amazonaws.com/cornet-models/cornet_s-1d3f7974.pth'
ckpt_data = torch.utils.model_zoo.load_url(url)
cornet.load_state_dict(ckpt_data['state_dict'])

# this FAnet is what we are going to train
realnet = CORnet_S().to(device)
realnet = torch.nn.DataParallel(realnet)
url = f'https://s3.amazonaws.com/cornet-models/cornet_s-1d3f7974.pth'
ckpt_data = torch.utils.model_zoo.load_url(url)
realnet.load_state_dict(ckpt_data['state_dict'])


class Data4Model(torch.utils.data.Dataset):
    def __init__(self, state='training', sub_index=1, transform=None):
        
        super(Data4Model, self).__init__()
        
        imgs = np.load('GetData/'+state+'_imgpaths.npy').tolist()
        
        if state=='training':
            n = 16540
        else:
            n = 200
        
        mean = np.load('GetData/preprocessed_mean_overall.npy')
        std = np.load('GetData/preprocessed_std_overall.npy')
        eeg = np.load('preprocessed_eeg_data/sub-'+str(sub_index).zfill(2)+'_'+state+'.npy')
        eeg = (eeg-mean[sub_index-1])/std[sub_index-1]
        
        self.imgs = imgs
        self.eeg = eeg
        self.transform = transform
  
    def __len__(self):
        return len(self.imgs)
  
    def __getitem__(self, item):
        imgs = self.transform(Image.open(self.imgs[item]).convert('RGB'))
        eeg = torch.tensor(self.eeg[item]).float()
         
        return imgs, eeg


task_criterion = nn.CrossEntropyLoss()

mse_criterion = nn.MSELoss()

class Gen_criterion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, eeg, criterion):
    
        loss1 = criterion(pred, eeg)
    
        pos_corr = []
        neg_corr = []
        n = pred.shape[0]
        for i in range(n):
            for j in range(n):
                if i == j:
                    pos_corr.append(spearman_corrcoef(pred[i], eeg[j]))
                else:
                    neg_corr.append(spearman_corrcoef(pred[i], eeg[j]))
        loss2 = 1 - torch.mean(torch.tensor(pos_corr)) + torch.mean(torch.tensor(neg_corr))
        loss = loss1 + loss2
        return loss

gen_criterion = Gen_criterion()


def train_and_test(encoder, cornet, weightspath, task_criterion, mse_criterion, gen_criterion, optimizer, transform,
                   beta=100, sub_index=1, batchsize=64, num_epochs=100):

    train_dataset = Data4Model(state='training', sub_index=sub_index, transform=transform)
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=batchsize, shuffle=True)
    
    test_dataset = Data4Model(state='test', sub_index=sub_index, transform=transform)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=batchsize, shuffle=False)
    
    since = time.time()
    
    loss_save = np.zeros([num_epochs, 6])

    best_model_params_path = os.path.join(weightspath + 'best_model_params.pt')
    
    cornet.eval()    

    best_corr = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
    
        # Training Session
    
        encoder.train()
            
        running_loss1 = 0.0
        running_loss2 = 0.0
        running_loss = 0.0
    
        # Iterate over data.
        
        niterates = 0

        for imgs, eeg in tqdm(train_data_loader):
            
            imgs = imgs.to(device)
            eeg = eeg.to(device)
        
            # zero the parameter gradients
            optimizer.zero_grad()
        
            # forward
            outputs, pred = encoder(imgs)
            cornet_outputs = cornet(imgs)
            loss1 = mse_criterion(outputs, cornet_outputs)
            loss2 = gen_criterion(pred, eeg, mse_criterion)
            loss = beta*loss2 + loss1
        
            # backward + optimize
            loss.backward()
            optimizer.step()
            
            running_loss1 += loss1.item()
            running_loss2 += loss2.item()
            running_loss += loss.item()
            
            niterates += 1
        
        loss_save[epoch, 0] = running_loss1/niterates
        loss_save[epoch, 1] = running_loss2/niterates
        loss_save[epoch, 2] = running_loss/niterates
            
        print(f'Train Loss: {running_loss/niterates:.4f} Task Loss: {running_loss1/niterates:.4f} Enc Loss: {running_loss2/niterates:.4f}')
    
        # Test Session
    
        encoder.eval()
        
        running_loss1 = 0.0
        running_loss2 = 0.0
        running_loss = 0.0
    
        # Iterate over data.
        
        niterates = 0

        for imgs, eeg in tqdm(test_data_loader):
            
            imgs = imgs.to(device)
            eeg = eeg.to(device)
        
            # zero the parameter gradients
            optimizer.zero_grad()
        
            # forward
            outputs, pred = encoder(imgs)
            cornet_outputs = cornet(imgs)
            loss1 = mse_criterion(outputs, cornet_outputs)
            loss2 = gen_criterion(pred, eeg, mse_criterion)
            loss = beta*loss2 + loss1
            
            running_loss1 += loss1.item()
            running_loss2 += loss2.item()
            running_loss += loss.item()
            
            niterates += 1
        
        loss_save[epoch, 3] = running_loss1/niterates
        loss_save[epoch, 4] = running_loss2/niterates
        loss_save[epoch, 5] = running_loss/niterates
            
        print(f'Test Loss: {running_loss/niterates:.4f} Task Loss: {running_loss1/niterates:.4f} Enc Loss: {running_loss2/niterates:.4f}')
        
        epoch_loss = running_loss/niterates
        
        if epoch == 0:
            best_loss = epoch_loss
        
        # deep copy the model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(encoder.state_dict(), best_model_params_path)
            
        epoch_model_params_path = os.path.join(weightspath + 'epoch'+str(epoch)+'_model_params.pt')
        torch.save(encoder.state_dict(), epoch_model_params_path)

    time_elapsed = time.time() - since
    
    np.savetxt(weightspath + 'loss.txt', loss_save)
    
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best test Loss: {best_loss:4f}')

    
# to train 10 ReAlnets based on 10 subjects' EEG data
for i in range(10):
    
    set_seed(2023)
    
    encoder = Encoder(realnet, 340).to(device)
    
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.000002)
    train_and_test(encoder, cornet, '/weights/ReAlnet_EEG/sub-'+str(i+1).zfill(2)+'/',
                   task_criterion, mse_criterion, gen_criterion, optimizer, transform, beta=100,
                   sub_index=i+1, batchsize=16, num_epochs=25)




