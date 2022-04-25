from __future__ import print_function, division
import os

import random
import pandas as pd
import numpy as np


import torch
import torchvision.models as models
import torch.utils.data
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from sklearn.metrics import accuracy_score

from model import combine_feature_model, muin_model, img_model
from Data_Loader import Load_Com_Data, load_combined_xy
from utils import check_create_dir

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
pin_memory = True

# train setting
epoch = 150
batch_size = 64
print('epoch = ' + str(epoch))
print('batch_size = ' + str(batch_size))

SEED = 25
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

valid_loss_min = 100
acc_max = 0
#acc_max = 0.793


lossT = []
lossL = []
lossL.append(np.inf)
lossT.append(np.inf)
epoch_valid = epoch - 2
n_iter = 1
i_valid = 0

#output_root = os.getcwd()
#######################################################
# Passing the Dataset of Images and Labels
#######################################################


data_root = '/main/data/Prediction'

feature_names = ['HB', 'drug-new', 'serum', 'AST', 'CP', 'gender', 'age', 'ALB', 'PT', 'ALT', 'CPR', 'AFP']

img_root ='/main/data/Prediction'
img_dir = os.path.join(img_root, 'pre_img')
mask_dir = os.path.join(img_root, 'gts')
multi_dir = os.path.join(img_root, 'multi_gts')
# 50
#input_channel = 1

# train loader
train_data_dir = os.path.join(data_root, 'TREATtr.xlsx') # patients' clinical data
original_train_data = pd.read_excel(train_data_dir)
train_img_path, train_mask_path, train_multi_path, x_train, y_train, input_channels = load_combined_xy(img_dir, mask_dir, multi_dir, original_train_data, feature_names)
Train_data = Load_Com_Data(train_img_path, train_mask_path, train_multi_path, x_train, y_train, 'train')

num_train = len(Train_data)
print(num_train)

# test loader
val_data_dir = os.path.join(data_root, 'TREATte.xlsx') # patients' clinical data
original_val_data = pd.read_excel(val_data_dir)
val_img_path, val_mask_path, val_multi_path, x_val, y_val, input_channels2 =load_combined_xy(img_dir, mask_dir, multi_dir, original_val_data, feature_names)

Val_data = Load_Com_Data(val_img_path, val_mask_path, val_multi_path, x_val, y_val, 'val')
num_val = len(Val_data)
print(num_val)


dataloaders_dict = {
    'train': torch.utils.data.DataLoader(Train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory),
    'val': torch.utils.data.DataLoader(Val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)}

if input_channels == input_channels2:
    ic = input_channels
else: 
    exit(0)
print (ic)

#######################################################
# Loading model
#######################################################
mlp = combine_feature_model(ic)
#mlp_path ='/main/models/Prediction/model1.path'
#mlp.load_state_dict(torch.load(mlp_path, map_location={'cuda:7':'cuda:7'}))
model1 = mlp.to(device)


rn = models.resnet18(pretrained=False)
resnet = img_model(rn, 0.2)
#resnet_path = '/main/models/Prediction/model2.path'
#resnet.load_state_dict(torch.load(resnet_path, map_location = {'cuda:7':'cuda:7'}))
model2 = resnet.to(device)

model3 = muin_model(1128)
#muin_path = '/main/models/Prediction/model3.path'
#model3.load_state_dict(torch.load(muin_path, map_location = {'cuda:7':'cuda:7'}))
model3 = model3.to(device)



#######################################################
# Optimizer setting
#######################################################
criterion = torch.nn.CrossEntropyLoss()




initial_lr = 1e-3

params =([{"params":model3.parameters(), "lr": initial_lr},
          {"params":model2.parameters(), "lr": initial_lr/100},
          {"params":model1.parameters(), "lr": initial_lr/100},
        ])
optimizer = torch.optim.Adam(model.parameters(), lr = initial_lr)

MAX_STEP = int(1e10)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, MAX_STEP, eta_min=1e-8)


#tried different otimizer setting

#optimizer = torch.optim.SGD(params, lr = initial_lr, momentum=0.8)
# STEPS_PER_EPOCH = len(dataloaders_dict['train'])
# TOTAL_STEPS = epoch * STEPS_PER_EPOCH

# MAX_LRS = [p['lr'] for p in optimizer.param_groups]
# #
# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
#                                                 max_lr=MAX_LRS,
#                                                 total_steps=TOTAL_STEPS)
# MAX_STEP = int(1e10)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 25, gamma=0.1, last_epoch=-1)





#######################################################
# output_root
#######################################################
output_root = './exp2'
check_create_dir(output_root)

#######################################################
# Writing the params to tensorboard
#######################################################
writer_dir = os.path.join(output_root, 'board')
check_create_dir(writer_dir)

writer1 = SummaryWriter(writer_dir)

#######################################################
# Creating a Folder for every data of the program
#######################################################
models_dir1 = os.path.join(output_root, 'model1')
check_create_dir(models_dir1)

models_dir2 = os.path.join(output_root, 'model2')
check_create_dir(models_dir2)

models_dir3 = os.path.join(output_root, 'model3')
check_create_dir(models_dir3)

#######################################################s
# Training
#######################################################
flag = False
train_losses = []
train_acces = []
val_losses = []
val_acces = []

for i in range(epoch):

    train_loss = 0.0
    valid_loss = 0.0

    #######################################################
    # Training Data
    #######################################################
    phase = 'train'
# tried to train models desperately

#    model1.eval()
#    model2.eval()
#    model3.eval()
    model1.train()
    model2.train()
    model3.train()

    for x, f, y in dataloaders_dict[phase]:

        x, f, y = x.to(device), f.to(device), y.to(device)

        optimizer.zero_grad()
        out1 = model1(x, flag)
        out2 = model2(f, flag)
        input_x = torch.cat((out1, out2), 1)
        
        y_pred = model3(input_x)
        
        lossT = criterion(y_pred, y)
        f_loss = lossT

        f_loss.backward()
        optimizer.step()

        train_loss += f_loss.item() * y.size(0)
        y_size = f_loss.item() * y.size(0)


    scheduler.step()
    train_losses.append(train_loss)

    #######################################################
    # Validation Step
    #######################################################
    phase = 'val'
    model1.eval()
    model2.eval()
    model3.eval()
    gt = []
    pred = []
    accuracy = 0.0
    with torch.no_grad():
        for x1, f1, y1  in dataloaders_dict[phase]:
            x1, f1, y1 = x1.to(device), f1.to(device), y1.to(device)
            out1_1 = model1(x1, flag)
            out2_1 = model2(f1, flag)
            input_x1 = torch.cat((out1_1, out2_1), 1)
            y_pred_1 = model3(input_x1)
            
            lossT = criterion(y_pred_1, y1)

            f_loss = lossT

            valid_loss += f_loss.item() * y1.size(0)
            y1_size = f_loss.item() * y1.size(0)
            y_prob = F.softmax(y_pred_1, dim=-1)
            top_pred = y_prob.argmax(1, keepdim=True)
            gt.append(y1.cpu())
            pred.append(top_pred.cpu())


    val_losses.append(valid_loss)
    gt = torch.cat(gt, dim=0)
    pred = torch.cat(pred, dim=0)
    accuracy = accuracy_score(gt, pred)

    #######################################################
    # To write in Tensorboard
    #######################################################

    train_loss = train_loss / num_train
    valid_loss = valid_loss / num_val
    writer1.add_scalars('Train_val_loss', {'train_loss': train_loss}, i)
    writer1.add_scalars('Train_val_loss', {'val_loss': valid_loss}, i)

    if (i + 1) % 1 == 0:
        print('Epoch: {}/{} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(i + 1, epoch, train_loss,valid_loss))

    #######################################################
    # Early Stopping
    #######################################################

    if valid_loss <= valid_loss_min and epoch_valid >= i:  # and i_valid <= 2:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model '.format(valid_loss_min, valid_loss))

        model_name = 'loss_min.path'
        model_path1 = os.path.join(models_dir1, model_name)
        torch.save(model1.state_dict(), model_path1)

        model_path2 = os.path.join(models_dir2, model_name)
        torch.save(model2.state_dict(), model_path2)

        model_path3 = os.path.join(models_dir3, model_name)
        torch.save(model3.state_dict(), model_path3)

        if round(valid_loss, 4) == round(valid_loss_min, 4):
            print(i_valid)
            i_valid = i_valid + 1
        valid_loss_min = valid_loss

    if acc_max <= accuracy and epoch_valid >= i:  # and i_valid <= 2:
        print('Accuracy increased ({:.6f} --> {:.6f}).  Saving model '.format(acc_max, accuracy))

        model_name = 'acc_max.path'
        model_path1 = os.path.join(models_dir1, model_name)
        torch.save(model1.state_dict(), model_path1)

        model_path2 = os.path.join(models_dir2, model_name)
        torch.save(model2.state_dict(), model_path2)

        model_path3 = os.path.join(models_dir3, model_name)
        torch.save(model3.state_dict(), model_path3)

        if round(acc_max, 4) == round(accuracy, 4):
            print(i_valid)
            i_valid = i_valid + 1
        acc_max = accuracy
#######################################################
# closing the tensorboard writer
#######################################################

writer1.close()


