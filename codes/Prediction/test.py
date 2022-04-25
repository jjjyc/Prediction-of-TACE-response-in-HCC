from __future__ import print_function, division
import os


import random
import pandas as pd
import numpy as np
import torch

import torch.utils.data
from sklearn.metrics import accuracy_score
import torch.nn.functional as F

from Data_Loader import Load_Com_Data, load_combined_xy

import torchvision.models as models
from model import combine_feature_model, muin_model, img_model

#######################################################
# Setting GPU
#######################################################

device = torch.device("cuda:7")
print(device)

#######################################################
# Setting the basic paramters of the model
#######################################################
# data loader
num_workers = 4
pin_memory = False
# train setting

batch_size = 64

print('batch_size = ' + str(batch_size))

SEED = 25
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

#######################################################
# Passing the Dataset of Images and Labels
#######################################################
data_root = '/main/data/Prediction'

feature_names = ['HB', 'drug-new', 'serum', 'AST', 'CP', 'gender',	'age', 'ALB', 'PT', 'ALT', 'CPR', 'AFP']
img_root = '/main/data/Prediction'
img_dir = os.path.join(img_root, 'pre_img')
mask_dir = os.path.join(img_root, 'gts')
multi_dir = os.path.join(img_root, 'multi_gts')

# test loader
val_data_dir = os.path.join(data_root, 'TREATte.xlsx')
original_val_data = pd.read_excel(val_data_dir)
val_img_path, val_mask_path, val_multi_path, x_val, y_val, input_channels2 =load_combined_xy(img_dir, mask_dir, multi_dir, original_val_data, feature_names)

Test_data = Load_Com_Data(val_img_path, val_mask_path, val_multi_path, x_val, y_val, 'val')
num_test = len(Test_data)
print(num_test)

dataloaders_dict = {'val': torch.utils.data.DataLoader(Test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)}


#######################################################
# Loading model
#######################################################

model_root = '/main/models/Prediction'


mlp = combine_feature_model(50)
mlp_path = model_root + '/model1.path'
mlp.load_state_dict(torch.load(mlp_path, map_location={'cuda:7':'cuda:7'}))
model1 = mlp.to(device)


rn = models.resnet18(pretrained=False)
resnet = img_model(rn, 0.2)

resnet_path = model_root + '/model2.path'
resnet.load_state_dict(torch.load(resnet_path, map_location = {'cuda:7':'cuda:7'}))
model2 = resnet.to(device)

model3 = muin_model(1128)
muin_path = model_root + '/model3.path'
model3.load_state_dict(torch.load(muin_path, map_location = {'cuda:7':'cuda:7'}))
model3 = model3.to(device)

model1.eval()
model2.eval()
model3.eval()

criterion = torch.nn.CrossEntropyLoss()

#######################################################
# Test Step
#######################################################
T_loss = 0
flag = False
gt = []
pred = []
with torch.no_grad():
    for x, f, y in dataloaders_dict['val']:
        x, f, y = x.to(device), f.to(device), y.to(device)
        out1 = model1(x, flag)
        out2 = model2(f, flag)
        input_x = torch.cat((out1, out2), 1)
        
        y_pred = model3(input_x)

        f_loss = criterion(y_pred, y)
        
        T_loss += f_loss.item() * y.size(0)
        y.size = f_loss.item() * y.size(0)
        
        y_prob = F.softmax(y_pred, dim=-1)
        top_pred = y_prob.argmax(1, keepdim=True)
        gt.append(y.cpu())
        pred.append(top_pred.cpu())
     
gt = torch.cat(gt, dim=0)

pred = torch.cat(pred, dim=0)
loss = T_loss/num_test

accuracy = accuracy_score(gt, pred)
print ('test acc:', accuracy)
print ('test loss:', loss)

