import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

class ImageDataset(Dataset):
    def __init__(self,df,device):
        super().__init__()
        self.df=df
        self.device=device
    def __len__(self):
        return len(self.df)
    def __getitem__(self,x):
        image=torch.tensor(self.df.iloc[x]["image"],dtype=torch.float32,device=self.device).permute(2,0,1)
        label= torch.tensor(1 if self.df.iloc[x]["label"]=="vehicle" else 0,dtype=torch.long,device=self.device)
        return image,label
    
    
class conv_net(nn.Module):
    def __init__(self,nb_channels,classes,img_size=64):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels=nb_channels,out_channels=20,kernel_size=(3,3),padding=1)
        self.r1=nn.ReLU()
        self.pool1=nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        
        self.conv2=nn.Conv2d(in_channels=20,out_channels=50,kernel_size=(3,3),padding=1)
        self.r2=nn.ReLU()
        self.pool2=nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
    
        self.l1=nn.Linear(in_features=int((img_size/4)*(img_size/4)*50),out_features=500)
        self.r3=nn.ReLU()
        self.l2 = nn.Linear(in_features=500, out_features=classes)
        self.logSoftmax = nn.LogSoftmax(dim=1)
    def forward(self,x):
        
        out=self.conv1(x)
        out=self.r1(out)
        out=self.pool1(out)
        
        
        out=self.conv2(out)
        out=self.r2(out)
        out=self.pool2(out)
        
        #print(out.shape)
        out=torch.flatten(out,1)
        out=self.l1(out)
        out=self.r3(out)
        
        #print(out.shape)
        out=self.l2(out)
        out=self.logSoftmax(out)
        return out