"""Convolutional autoencoder components for nonlinear hyperspectral unmixing."""

import math
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import torch
import torch.nn.functional as F
import torch.utils.data
from barbar import Bar
from pysptools import eea
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from spectral import *
from torch import nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.tensorboard import SummaryWriter

def swish(x):
    """
    Compute the swish activation (x * sigmoid(x)).

    Parameters
    ----------
    x : torch.Tensor
        Tensor that will be transformed element-wise.

    Returns
    -------
    torch.Tensor
        Activated tensor with the same shape as ``x``.
    """
    return x * torch.sigmoid(x)


def spectral_div(x,y):
    """
    Compute the spectral information divergence between two spectra.

    Parameters
    ----------
    x : torch.Tensor
        First batch of spectra with shape ``(batch, wavelength)``.
    y : torch.Tensor
        Second batch of spectra with the same shape as ``x``.

    Returns
    -------
    torch.Tensor
        Scalar tensor containing the divergence value.
    """
    x = torch.clip(x,min=0.001)
    y = torch.clip(y,min=0.001)
    sum_x=torch.sum(x,dim=1)
    sum_y=torch.sum(y,dim=1)
    pn=(x.T/sum_x).T
    qn=(y.T/sum_y).T
    loss= torch.sum(pn*torch.log2(pn/qn)+qn*torch.log2(qn/pn))
    return loss


def spectral_distance(x,y):
    """
    Compute the average angular error between spectra.

    Parameters
    ----------
    x : torch.Tensor
        First batch of spectra with shape ``(batch, wavelength)``.
    y : torch.Tensor
        Second batch of spectra with the same shape as ``x``.

    Returns
    -------
    torch.Tensor
        Scalar tensor representing the mean angular distance.
    """
    loss=torch.mean(torch.acos(torch.diag(torch.mm(x,y.T),0)/(torch.norm(x,dim=1)*torch.norm(y,dim=1))))
    return loss



def RMSELoss(yhat,y):
    """
    Compute the root mean squared error loss between predictions and targets.

    Parameters
    ----------
    yhat : torch.Tensor
        Predicted spectra.
    y : torch.Tensor
        Ground-truth spectra.

    Returns
    -------
    torch.Tensor
        Scalar tensor containing the RMSE.
    """
    return torch.sqrt(torch.mean((yhat-y)**2))



def MSELos(yhat,y):
    """
    Compute the mean squared error loss between predictions and targets.

    Parameters
    ----------
    yhat : torch.Tensor
        Predicted spectra.
    y : torch.Tensor
        Ground-truth spectra.

    Returns
    -------
    torch.Tensor
        Scalar tensor containing the MSE.
    """
    return torch.mean((yhat-y)**2)


class ConvAutoencoder(nn.Module):
    """
    Convolutional autoencoder used for nonlinear hyperspectral unmixing.

    The encoder progressively downsamples the input spectrum to estimate
    abundances while the decoder reconstructs both linear and nonlinear
    components using the learned abundances and the provided endmember bases.
    """
    def __init__(self, nb_channel, basis, nb_endm):
        """
        Parameters
        ----------
        nb_channel : int
            Number of wavelength samples in each spectrum.
        basis : torch.Tensor
            Tensor of shape ``(nb_channel, nb_endm)`` containing the endmember
            signatures used by the decoder.
        nb_endm : int
            Number of endmembers (i.e., abundances) to predict.
        """
        super(ConvAutoencoder, self).__init__()
        self.basis=basis
        self.nb_channel = nb_channel
        self.nb_endm = nb_endm
        self.conv1 = nn.Conv1d(1,16,3,stride=2)
        self.max1 = nn.MaxPool1d(kernel_size=2,stride=1)
        self.batch1=nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16,32,3, padding=0,stride=2)
        self.max2 = nn.MaxPool1d(kernel_size=2,stride=1)
        self.batch2=nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32,64,3, stride=2)
        self.max3 = nn.MaxPool1d(kernel_size=2,stride=1)
        self.batch3=nn.BatchNorm1d(64)
        test_size=(((((self.max3(self.conv3(self.max2(self.conv2(self.max1(self.conv1(torch.rand(1,1,self.nb_channel))))))))))))
        
        self.convll1= nn.Conv1d(64, nb_endm,math.floor(test_size.shape[2]))
        self.batch6=nn.BatchNorm1d(nb_endm)
        self.convll2= nn.Conv1d(nb_endm, nb_endm,test_size.shape[2]-math.floor(test_size.shape[2]/2))
        self.ll1=nn.Linear(test_size.shape[1]*test_size.shape[2],256,bias=True)
        self.ll2=nn.Linear(256,nb_endm,bias=True)
        self.drop=nn.Dropout(p=0.09)
        self.weight=  nn.Parameter((self.basis.T))

        self.convnn= nn.Conv2d(1, 32,(5,nb_endm), stride=(1,1))
        self.convnn1= nn.Conv1d(32, 32,5, stride=1)
        self.convnn2= nn.Conv1d(32, 32,5, stride=1)
        test=(self.convnn(torch.rand(1,1,nb_channel,nb_endm)))
        test_=(self.convnn2(self.convnn1(test.reshape(-1,test.shape[1],test.shape[2]))))
        self.convnn3= nn.Linear(test_.shape[1]*test_.shape[2], nb_channel,bias=True)
        
    def forward(self, x,linear,abundances,nonlinear):
        """
        Run the encoder and decoder to produce abundances and reconstructions.

        Parameters
        ----------
        x : torch.Tensor
            Input spectra with shape ``(batch, 1, nb_channel)``.
        linear : bool
            If ``True`` return only the linear reconstruction.
        abundances : bool
            If ``True`` return the abundance vector.
        nonlinear : bool
            If ``True`` return only the nonlinear reconstruction.

        Returns
        -------
        torch.Tensor
            Abundances, linear component, nonlinear component, or their sum
            depending on the provided boolean flags.
        """
        x = (self.conv1(x))
        x = F.relu(self.max1(x))
        x = (self.conv2(x))
        x = F.relu(self.max2(x))
        x = (self.conv3(x))
        x = F.relu(self.max3(x))
        
        x =  nn.Flatten()(x)
        x = (self.ll1(x))
        x = (self.ll2(x))
        x_ =  nn.Flatten()(x)
        
        x = torch.abs(x_)

        inter=(torch.unsqueeze(x,1)*self.weight.T)
        
        xlin_=torch.sum(inter,2)
        
        array=torch.unsqueeze(inter,1)
        x2_= (self.convnn(array))
        x2_=x2_.reshape(-1,x2_.shape[1],x2_.shape[2])
        x2_= (self.convnn1(x2_))
        x2_= (self.convnn2(x2_))
        
        x2_=x2_.reshape(-1,x2_.shape[1]*x2_.shape[2])
        x2_= (self.convnn3(x2_))
        if abundances:
            return x
        if nonlinear:
            return x2_
        if linear:
            return xlin_
        else:
            return (xlin_+x2_)
        


def train_nn(n_epochs,model,optimizer,train_loader,valid_loader,nonlinear_weights_lamda,smoothing,path,name,device,linear):
    """
    Train the autoencoder while tracking training/validation losses.

    Parameters
    ----------
    n_epochs : int
        Number of optimization epochs.
    model : ConvAutoencoder
        Autoencoder model to optimize.
    optimizer : torch.optim.Optimizer
        Optimizer configured with the model parameters.
    train_loader : torch.utils.data.DataLoader
        Loader yielding training spectra.
    valid_loader : torch.utils.data.DataLoader
        Loader yielding validation spectra.
    nonlinear_weights_lamda : float
        Weight applied to the nonlinear regularization penalty.
    smoothing : float
        Weight applied to the endmember smoothing penalty.
    path : str
        Directory used to store tensorboard logs.
    name : str
        Run name (kept for backwards compatibility).
    device : torch.device
        Target device for training/inference.
    linear : bool
        Whether to request linear outputs from the model during optimization.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Arrays containing the per-epoch training and validation losses.
    """
    writer = SummaryWriter(log_dir=os.path.join(path,'Tensorboard'))
    Loss = nn.MSELoss(reduction='sum')
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.8)
    train_loss = []
    valid_loss = []
    epoch=0
    valid_res = np.inf
    while epoch<n_epochs:
        loss_t=0
        for i, (p) in tqdm.tqdm(enumerate(train_loader), total=int(len(train_loader))):
            x = Variable(p).float().to(device)
            y = Variable(p).float().to(device)
            optimizer.zero_grad()
            outputs= model(x,linear,False,False)
            loss=torch.tensor(0.)
            loss_distance = spectral_distance(outputs.squeeze(), torch.squeeze(y))
            loss_mse=Loss(outputs, torch.squeeze(y))
            loss_div=spectral_div(outputs.squeeze(), torch.squeeze(y))
            mse_lamda=1
            distance_lamda=1e-2
            divergence_lamda=0.7
            l2_reg = torch.tensor(0.).to(device)
            l1_reg = torch.tensor(0.).to(device)
            for h in range(model.weight.shape[0]):
                        l2_reg +=torch.sum(torch.abs(model.weight[h,1:]-model.weight[h,:-1]))
            non_linear_comp=torch.norm(model.convnn3.weight)
            
            loss =  loss_div +nonlinear_weights_lamda*non_linear_comp+smoothing*l2_reg
            model.weight.data=model.weight.data.clamp(0)
            loss.backward()
            optimizer.step()
            loss_t+=loss.item()
            
            
        valid = 0.0
        model.eval()
        for data in valid_loader:
            data = data.float().to(device)
            target = model(data,linear,False,False)
            loss = spectral_div(target,data.squeeze())
            valid += loss.item()
        scheduler.step(valid)
        if valid/(len(valid_loader))<valid_res:
           valid_res=valid/(len(valid_loader))
           torch.save(model.state_dict(), os.path.join(path,name))
           print('Saving')   
        print('Epoch :{}, Lr :{}, Loss:{}, val_loss:{}'.format(epoch,optimizer.param_groups[1]['lr'],loss_t/len(train_loader),valid_res))
        writer.add_scalar("Loss/Train", loss_t/len(train_loader) , epoch)
        writer.add_scalar("Loss/Validation", valid_res , epoch)
        fig=plt.figure()
        plt.plot(model.weight.cpu().detach().numpy().T)
        writer.add_figure('Basis',fig,global_step=epoch)
        epoch=epoch+1 
        train_loss.append(loss_t/len(train_loader))
        valid_loss.append(valid_res)
        time.sleep(0.3)
    return np.array(train_loss),np.array(valid_loss)

def get_abundances(model,test_loader,device,linear):
    """
    Estimate the abundance vectors for the provided spectra.

    Parameters
    ----------
    model : ConvAutoencoder
        Trained model used for inference.
    test_loader : torch.utils.data.DataLoader
        Loader providing spectra from which to extract abundances.
    device : torch.device
        Device on which inference is executed.
    linear : bool
        Whether to request the linear reconstruction in addition to abundances.

    Returns
    -------
    numpy.ndarray
        Array containing the stacked abundance predictions.
    """
    model.eval()
    for i, (x) in enumerate(test_loader):
        x = Variable(x).float().to(device)
        if i==0:
           outputs= model(x,linear,True,True)
           outputs=outputs.cpu().detach().numpy()
        else:
          a=model(x,linear,True,True)
          a=a.cpu().detach().numpy()
          outputs=np.concatenate((outputs,a))
        del(x)
    return outputs


def decode(model,abundances,device):
    """
    Reconstruct spectra from abundances using the trained decoder branch.

    Parameters
    ----------
    model : ConvAutoencoder
        Trained autoencoder whose decoder will be reused.
    abundances : torch.Tensor or numpy.ndarray
        Abundance matrix with shape ``(samples, nb_endm)``.
    device : torch.device
        Device on which to run the decoder.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Linear reconstruction, nonlinear reconstruction, and their sum.
    """

    def decoder(model,abundances):
        """Decode a batch of abundances into linear, nonlinear, and combined outputs."""
        model=model.to(device)
        inter=(torch.unsqueeze(abundances,1)*model.weight.T)
        xlin_=torch.sum(inter,2)
        array=torch.unsqueeze(inter,1)
        x2_= (model.convnn(array))
        x2_=x2_.reshape(-1,x2_.shape[1],x2_.shape[2])
        x2_= (model.convnn1(x2_))
        x2_= (model.convnn2(x2_))
        x2_=x2_.reshape(-1,x2_.shape[1]*x2_.shape[2])
        x2_= (model.convnn3(x2_))
        return xlin_,x2_,xlin_+x2_
    
    test_loader = torch.utils.data.DataLoader(abundances, batch_size=1024, shuffle=False)
    model.eval()
    for i, (x) in enumerate(test_loader):
        x = Variable(x).float().to(device)
        if i==0:
            x1,x2,x3= decoder(model,x)
            x1,x2,x3=x1.cpu().detach().numpy(),x2.cpu().detach().numpy(),x3.cpu().detach().numpy()
        else:
            x1_,x2_,x3_= decoder(model,x)
            x1_,x2_,x3_=x1_.cpu().detach().numpy(),x2_.cpu().detach().numpy(),x3_.cpu().detach().numpy()
            x1=np.concatenate((x1,x1_))
            x2=np.concatenate((x2,x2_))
            x3=np.concatenate((x3,x3_))
        del(x)
    return x1,x2,x3


def clean_image(image,bands,threshold_min,threshold_max,swir=True,bands_min=1136,bands_max=2413):
    """
    Remove noisy spectral bands and extreme pixels from an HSI cube.

    Parameters
    ----------
    image : numpy.ndarray
        Hyperspectral cube in ``(rows, cols, bands)`` or ``(pixels, bands)``.
    bands : numpy.ndarray
        Center wavelength of each band.
    threshold_min : float
        Lower quantile threshold (between 0 and 1) used to mask dark pixels.
    threshold_max : float
        Upper quantile threshold used to mask overly bright pixels.
    swir : bool, default True
        If ``True`` clip bands outside ``[bands_min, bands_max]``.
    bands_min : int, default 1136
        Minimum wavelength to keep when ``swir`` is ``True``.
    bands_max : int, default 2413
        Maximum wavelength to keep when ``swir`` is ``True``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Filtered image and a boolean mask marking removed pixels.
    """
    if swir:
        noise1=(bands<bands_min)
        noise2=(bands>bands_max)
        combined=noise1 | noise2
    else:
        noise1=(bands<450)
        combined=noise1
        
    if len(image.shape)>2:
        image=image[:,:,~combined]
        dim1,dim2,dim3=image.shape
        norm=np.linalg.norm(image.reshape(-1,dim3),axis=1,ord=2)
    else:
        image=image[:,~combined]
        dim1,dim2=image.shape
        norm=np.linalg.norm(image.reshape(-1,dim2),axis=1,ord=2)
    threshold1=np.quantile(norm,threshold_max)
    threshold2=np.quantile(norm,threshold_min)
    mask=(norm>threshold1) | (norm<threshold2)
    if len(image.shape)>2:
        mask=mask.reshape(dim1,dim2)
    return image,mask
    
