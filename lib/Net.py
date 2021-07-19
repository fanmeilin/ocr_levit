import torch.nn as nn
import timm

class ViTBase16(nn.Module):
    def __init__(self,n_classes,pretrained = False):
        super(ViTBase16,self).__init__()
        self.model = timm.create_model("resnet34",pretrained=pretrained,in_chans =1,num_classes=n_classes)
    def forward(self,x):
        x = self.model(x)
        x = x.argmax(dim=1)
        return x