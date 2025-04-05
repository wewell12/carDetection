import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn


class BigImageDataset(Dataset):
    def __init__(self,df,device):
        super().__init__()
        self.df=df
        self.device=device
    def __len__(self):
        return len(self.df)
    def __getitem__(self,x):
        image=torch.tensor(self.df.iloc[x]["image"],dtype=torch.float32,device=self.device)
        #label= torch.tensor(1 if self.df.iloc[x]["label"]=="vehicle" else 0,dtype=torch.long,device=self.device)
        return image
    
    
    
def reduce_tensor(t,size,device):
    shape=[t.shape[0],t.shape[1]]
    step=[shape[0]//size[0],shape[1]//size[1]]
    resized_image=torch.empty((*size,3),dtype=torch.float32,device=device)
    for i in range(0,shape[0],step[0]):
        for j in range(0,shape[1],step[1]):
            r=t[i:i+step[0]].permute(1,0,2)
            r=r[j:j+step[1]].permute(1,0,2)
            resized_image[i//step[0]][j//step[1]]=torch.sum(r,axis=[0,1])/(step[0]*step[1])
    return resized_image

def inflate_tensor(t,size,device):
    shape=[t.shape[0],t.shape[1]]
    step=[size[0]//shape[0],size[1]//shape[1]]
    resized_image=None
    for i in range(shape[0]):
        line=None
        for j in range(shape[1]):
        
            res=torch.full((*step,),t[i][j][0].item(),dtype=torch.float32,device=device).unsqueeze(0)
            res=torch.vstack((res,torch.full((*step,),t[i][j][1],dtype=torch.float32,device=device).unsqueeze(0)))
            res=torch.vstack((res,torch.full((*step,),t[i][j][2],dtype=torch.float32,device=device).unsqueeze(0)))
            res=res.permute(1,2,0)
            if(j==0):
                line=res
            else :
                line=torch.hstack((line,res))
        if(i==0):
            resized_image=line
        else :
            resized_image=torch.vstack((resized_image,line))
                
    return resized_image


def edge_regulation(f,w_size):
    res=np.full((f.shape),w_size*w_size)
        
    for i in range(len(f)):
        if i<w_size :
            res[i]=[w_size*(i+1)]*len(f[0])
            res[len(f)-i-1]=[w_size*(i+1)]*len(f[0])
        
            for j in range(w_size):
                res[i][j]=(i)*(j)+1
                res[i][len(f[0])-j-1]=(i)*(j)+1
                res[len(f)-i-1][j]=(i)*(j)+1
                res[len(f)-i-1][len(f[0])-j-1]=(i)*(j)+1
        elif i<len(f)-w_size :
            for j in range(w_size):
                res[i][j]=w_size*j+1
                res[len(f)-i-1][len(f[0])-j-1]=w_size*j+1
                pass
            
    return f*(1/np.exp((res/(w_size*w_size))))


def check_in_images(dataset,model,size,step=8,device="cpu",edge_reg=True):
    result=[]
    for img in dataset :
        res_img=np.zeros((len(img),len(img[0])))
        if(len(img)>size[0] and len(img[0])>size[1]) :
            for offsety in range(0,len(img)-size[0]-1,step):
                for offsetx in range(0,len(img[0])-size[1]-1,step):
                    small_img=img[offsety:offsety+size[0]].permute(1,0,2)
                    small_img=small_img[offsetx:offsetx+size[1]].permute(1,0,2)
                    
                    if(size[0]>64):
                    
                        casted_img=reduce_tensor(small_img,[64,64],device)
                        
                    elif(size[0]<64):
                        casted_img=inflate_tensor(small_img,[64,64],device)
                    else :
                        casted_img=small_img
                    pred=model(casted_img.permute(2,0,1).unsqueeze(0))
                        
                    if(pred.argmax().item()==1):
                        for i in range(offsety,offsety+size[0]):
                            for j in range(offsetx,offsetx+size[1]):
                                res_img[i][j]+=1
            
            if edge_reg :
                result.append(edge_regulation(res_img,step))
            else :
                result.append(res_img)
                
    return result


def def_image_quant(filter_img,p=0.90):
    def_img=np.empty_like(filter_img)
    q=np.quantile(filter_img.flat,p)
    for i in range(len(def_img)):
        for j in range(len(def_img[0])):
            if(filter_img[i][j]>q):
                def_img[i][j]=1
            else :
                def_img[i][j]=0
    return def_img

def filter_regulation(f):
    r=np.max(f)
    return f/r

def def_image_value(filter_img,p=0.90):
    filter_img=filter_regulation(filter_img)
    def_img=np.empty_like(filter_img)
    for i in range(len(def_img)):
        for j in range(len(def_img[0])):
            if(filter_img[i][j]>p):
                def_img[i][j]=1
            else :
                def_img[i][j]=0
    return def_img

def combine_filter_image(image,f):
    def_img=np.empty_like(image)
    for i in range(len(def_img)):
        for j in range(len(def_img[0])):
            if(f[i][j]==1):
                def_img[i][j]=np.array([image[i][j][0],1,image[i][j][2]])
            else :
                def_img[i][j]=image[i][j]
    return def_img
    

