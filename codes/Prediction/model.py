import torch
import torch.nn as nn


class combine_feature_model(torch.nn.Module):
    #input = all features combined
    def __init__(self, feature_channel):
        super(combine_feature_model, self).__init__()

        self.relu = nn.ReLU()
  
        self.up = nn.Linear(feature_channel, 128)
      
        self.bm = nn.BatchNorm1d(128)

        self.down = nn.Linear(128, 2)
        self._initialize_weights()

    def forward(self, tab0, flag):
        
        x = self.up(tab0)
        x1 = self.relu(x)
        x1 = self.bm(x1)
        y = self.down(x1)

        if flag == False:
            return y
        else:
            return x1
        
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0, std=0.02)
                torch.nn.init.constant_(m.bias, 0.1)
 
class img_model(torch.nn.Module):
    def __init__(self, model_core, dd):
        super(img_model, self).__init__()
        
        self.resnet_model = model_core
        self.xfc = nn.Linear(1000, 1)
        nn.init.xavier_normal_(self.xfc.weight)

        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dd)
    def forward(self, x, flag):
       
        
        x0 = self.resnet_model(x)
        x1 = self.dropout(x0)

        x1 = x1.view(x1.size(0), -1)
        y1 = self.xfc(x1)

        if flag==False:
            return y1
        else:
            return x0

               
class muin_model(torch.nn.Module):
    def __init__(self, feature_channel):
        super(muin_model, self).__init__()

        self.down = nn.Linear(3, 2)

        self._initialize_weights()

    def forward(self, tab):
        
        x4 = self.down(tab)

        return x4
        
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0, std=0.02)
                torch.nn.init.constant_(m.bias, 0.1)
