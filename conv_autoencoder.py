import torch.utils.data
from torch.autograd import Variable
import torch
from torch import nn
import numpy as np
import scipy 
import os
import matplotlib.pyplot as plt
import pandas as pd
import random
import math
import tqdm
from sklearn.svm import SVC
from barbar import Bar
from pysptools import eea
import time
from spectral import*
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau,StepLR
import math
from torch.utils.tensorboard import SummaryWriter

def swish(x):
    """  swish activation function"""
    return x * torch.sigmoid(x)


def spectral_div(x,y):
    """  Spectral Information Divergence Loss """
    x = torch.clip(x,min=0.001)
    y = torch.clip(y,min=0.001)
    sum_x=torch.sum(x,dim=1)
    sum_y=torch.sum(y,dim=1)
    pn=(x.T/sum_x).T
    qn=(y.T/sum_y).T
    loss= torch.sum(pn*torch.log2(pn/qn)+qn*torch.log2(qn/pn))
    return loss


def spectral_distance(x,y):
    """ Spectral Distance Loss Function """
    loss=torch.mean(torch.acos(torch.diag(torch.mm(x,y.T),0)/(torch.norm(x,dim=1)*torch.norm(y,dim=1))))
    return loss



def RMSELoss(yhat,y):
    """ Root Mean Squared Error """
    return torch.sqrt(torch.mean((yhat-y)**2))



def MSELos(yhat,y):
    """  Mean Squared Error """
    return torch.mean((yhat-y)**2)


class ConvAutoencoder(nn.Module):
    """
    ConvAutoEncoder Class 
    
    Attributes
    ----------
    nb_channel : int
        specifies the number of wavelengths of a spectrum (usually 288)
        
    nb_endm : int
        specify number of endmembers to extract
        
    bases : torch tensor(nb_channel,nb_endm)
        Endmember Matrix
        
    abundanes: boolean
        if true return Abundances
        
    nonlinear: boolean
        if true return nonlinear output
    Returns
    ---------
    3 options depending on Abundances and nonlinear
    
    if abundances =True 
        return Abundances (batch_size,nb_endm)

    if nonlinear =True 
        return nonlinear output (batch_size,nb_channel)
     if nonlinear =True 
        return linear output (batch_size,nb_channel)
    if  both are False
        return reconstructed spectrum (linear+nonlinear)
    """
    def __init__(self, nb_channel, basis, nb_endm):
        super(ConvAutoencoder, self).__init__()
        self.basis=basis
        self.nb_channel = nb_channel
        # self.length = length
        self.nb_endm = nb_endm
        """ Layer 1"""
        # Conv Layer
        self.conv1 = nn.Conv1d(1,16,3,stride=2)
        # torch.nn.init.xavier_normal_(self.conv1.weight)
        # Maxpool Layer
        self.max1 = nn.MaxPool1d(kernel_size=2,stride=1)
        # BatchNormalization
        self.batch1=nn.BatchNorm1d(16)
        """ Layer 2"""
        # Conv Layer
        self.conv2 = nn.Conv1d(16,32,3, padding=0,stride=2)
        # torch.nn.init.xavier_normal_(self.conv2.weight)
        # Maxpool Layer
        self.max2 = nn.MaxPool1d(kernel_size=2,stride=1)
        # BatchNormalization
        self.batch2=nn.BatchNorm1d(32)
        """ Layer 3"""
        # Conv Layer
        self.conv3 = nn.Conv1d(32,64,3, stride=2)
        # torch.nn.init.xavier_normal_(self.conv3.weight)
        # Maxpool Layer
        self.max3 = nn.MaxPool1d(kernel_size=2,stride=1)
        # BatchNormalization
        self.batch3=nn.BatchNorm1d(64)
        """ Linear Layers"""
        test_size=(((((self.max3(self.conv3(self.max2(self.conv2(self.max1(self.conv1(torch.rand(1,1,self.nb_channel))))))))))))
        
        self.convll1= nn.Conv1d(64, nb_endm,math.floor(test_size.shape[2]))
        self.batch6=nn.BatchNorm1d(nb_endm)
        self.convll2= nn.Conv1d(nb_endm, nb_endm,test_size.shape[2]-math.floor(test_size.shape[2]/2))
        # torch.nn.init.xavier_normal_(self.convll1.weight)
        self.ll1=nn.Linear(test_size.shape[1]*test_size.shape[2],256,bias=True)
        # torch.nn.init.xavier_normal_(self.ll1.weight)
        self.ll2=nn.Linear(256,nb_endm,bias=True)
        # torch.nn.init.xavier_normal_(self.ll2.weight)
        # torch.nn.init.xavier_normal_(self.ll2.weight)
        """ Dropout """
        self.drop=nn.Dropout(p=0.09)
        """  Decoder"""
        self.weight=  nn.Parameter((self.basis.T))

        self.convnn= nn.Conv2d(1, 32,(5,nb_endm), stride=(1,1))
        # torch.nn.init.xavier_normal_(self.convnn.weight)
        self.convnn1= nn.Conv1d(32, 32,5, stride=1)
        # torch.nn.init.xavier_normal_(self.convnn1.weight)
        self.convnn2= nn.Conv1d(32, 32,5, stride=1)
        test=(self.convnn(torch.rand(1,1,nb_channel,nb_endm)))
        test_=(self.convnn2(self.convnn1(test.reshape(-1,test.shape[1],test.shape[2]))))
        self.convnn3= nn.Linear(test_.shape[1]*test_.shape[2], nb_channel,bias=True)
        # torch.nn.init.xavier_normal_(self.convnn2.weight)
        #..........................................
        
    def forward(self, x,linear,abundances,nonlinear):
        
        """ Encoder """
        x = (self.conv1(x))
        x = F.relu(self.max1(x))
        x = (self.conv2(x))
        x = F.relu(self.max2(x))
        x = (self.conv3(x))
        x = F.relu(self.max3(x))
        
        x =  nn.Flatten()(x)
        x = (self.ll1(x))
        x = (self.ll2(x))
        # print(x.shape)
        x_ =  nn.Flatten()(x)
        """  Absolute Value Rectification """
        
        x = torch.abs(x_)
        # x = (x.T/torch.sum(x,dim=1)).T
        # x = F.softmax(x_,1)
        
        """ Decoder """
        inter=(torch.unsqueeze(x,1)*self.weight.T)
        
        ### Linear Output
        xlin_=torch.sum(inter,2)
        
        ### Nonlinear Output
        array=torch.unsqueeze(inter,1)
        x2_= (self.convnn(array))
        x2_=x2_.reshape(-1,x2_.shape[1],x2_.shape[2])
        x2_= (self.convnn1(x2_))
        x2_= (self.convnn2(x2_))
        
        x2_=x2_.reshape(-1,x2_.shape[1]*x2_.shape[2])
        # print(x2_.shape)
        x2_= (self.convnn3(x2_))
        #..........................................
        """ Return Output """        
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
    Attributes
    ----------
    n_epochs: int
        number of full cycles over training data
    
    model: instance of ConvAutoEncoder 
    
    optimizer: An Algorithm of torch.optim  https://pytorch.org/docs/stable/optim.html
    
    train_loader: Torch Datalaoder for training data https://pytorch.org/docs/stable/data.html
    
    valid_loader: Torch Datalaoder for testing data
    
    nonlinear_weights_lamda: float
        power of nonlinear output (often around 1e-3)
    
    smoothing: float
        power of smoothing of endmembers (often around 1e-6)
    
    path: string
        path of folder where model dictionary state to be saved
        
    name: string
        name of saved model
        
    Returns
    -----------
    train_loss: numpy.array (n_epochs)
    valid_loss: numpy.array (n_epochs)
    
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
            # clear the gradients
            optimizer.zero_grad()
            # forward pass
            outputs= model(x,linear,False,False)
            # calculate the loss
            loss=torch.tensor(0.)
            loss_distance = spectral_distance(outputs.squeeze(), torch.squeeze(y))
            loss_mse=Loss(outputs, torch.squeeze(y))
            loss_div=spectral_div(outputs.squeeze(), torch.squeeze(y))
            mse_lamda=1
            distance_lamda=1e-2
            divergence_lamda=0.7
            l2_reg = torch.tensor(0.).to(device)
            l1_reg = torch.tensor(0.).to(device)
            # l1_reg +=torch.norm(model.weight2)
            l2_lambda = smoothing
            for h in range(model.weight.shape[0]):
                        l2_reg +=torch.sum(torch.abs(model.weight[h,1:]-model.weight[h,:-1]))
            non_linear_comp=torch.norm(model.convnn3.weight)
            
            loss =  loss_div +nonlinear_weights_lamda*non_linear_comp+smoothing*l2_reg
                # loss =  loss_mse +smoothing*l2_reg
                # backward pass
            model.weight.data=model.weight.data.clamp(0)
            loss.backward()
            # perform a single optimization step
            optimizer.step()
            loss_t+=loss.item()
            
            
        """  validation loss after each epoch"""
        valid = 0.0
        model.eval()     # Optional when not using Model Specific layer
        for data in valid_loader:
            data = data.float().to(device)
            # Forward Pass
            target = model(data,linear,False,False)
            # Find the Loss
            loss = spectral_div(target,data.squeeze())
            # Calculate Loss
            valid += loss.item()
        scheduler.step(valid)
        # optimizer.param_groups[0]["lr"]=optimizer.param_groups[0]["lr"]*0.98
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
        # scheduler.step()
        epoch=epoch+1 
        train_loss.append(loss_t/len(train_loader))
        valid_loss.append(valid_res)
        time.sleep(0.3)
    return np.array(train_loss),np.array(valid_loss)

def get_abundances(model,test_loader,device,linear):
    
    """"
    Attributes
    ----------
    model: trained version of ConvAutoEncoder
    
    test_loader: torch Dataloader with test data
    
    Returns
    -----------
    outputs: numpy.array (len(test_loader)*batch_size,nb_endm)
    
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
    """"
    Attributes
    ----------
    model: trained version of ConvAutoEncoder
    
    abundances: output of  get_abundances function

    device: torch.device()
    
    Returns
    -----------
    outputs: 
    
    x1: torch.tensor (abundances.shape[0],nb_channel)
        linear output
    x2: torch.tensor (abundances.shape[0],nb_channel)
        nonlinear output
    x3: torch.tensor (abundances.shape[0],nb_channel)
        linear + nonlinear output
    
    """

    def decoder(model,abundances):
        # x=torch.from_numpy(abundances).cpu()
        model=model.to(device)
        inter=(torch.unsqueeze(x,1)*model.weight.T)
        ### Linear Output
        xlin_=torch.sum(inter,2)
        ### Nonlinear Output
        array=torch.unsqueeze(inter,1)
        x2_= (model.convnn(array))
        x2_=x2_.reshape(-1,x2_.shape[1],x2_.shape[2])
        x2_= (model.convnn1(x2_))
        x2_= (model.convnn2(x2_))
        x2_=x2_.reshape(-1,x2_.shape[1]*x2_.shape[2])
        # print(x2_.shape)
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
    """"
    Attributes
    ----------
    image : numpy.array (rows,cols,bands)
    
    bands : numpy.array np.array(hdr.bands.centers)
    
    Return
    ---------
    image : numpy.array (rows,cols,new_bands)
    mask: numpy.array(rows,cols)
    
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
    