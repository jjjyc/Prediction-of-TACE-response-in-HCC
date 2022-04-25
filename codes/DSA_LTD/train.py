from __future__ import print_function, division

import os
import shutil
import random
import numpy as np

import torch.utils.data
from tensorboardX import SummaryWriter

from Metrics import dice_coeff
from unet import UNet
from losses import calc_loss
from Data_Loader import Images_Dataset_folder
from utils import check_create_dir


#######################################################
# Setting GPU
#######################################################

device = torch.device("cuda:1")
device1 = torch.device("cuda:2")
device2 = torch.device("cuda:3")
print (device)
print (device1)
print (device2)
#######################################################
# Setting the basic paramters of the model
#######################################################
pin_memory = True

epoch = 200
batch_size = 8

class_num = 0
input_channel0 = 10
input_channel1 = 1
input_channel2 = 3
output_channel = 1

lamda0 = 0.1
lamda1 = 1
lamda2 = 1

SEED = 25
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

valid_loss_min = np.Inf
dice_max = 0

lossT1 = []
lossT2 = []
lossL = []
lossL.append(np.inf)
lossT1.append(np.inf)
lossT2.append(np.inf)
epoch_valid = epoch - 2
n_iter = 1
i_valid = 0

print('epoch = ' + str(epoch))
print('batch_size = ' + str(batch_size))
print('random_seed = ' + str(SEED))
print('best_dice = ' + str(dice_max))

#######################################################
# Setting up the model
#######################################################


model_test0 = UNet(input_channel0, output_channel)  #model_TDL
model_test0.to(device)

model_test1 = UNet(input_channel1, output_channel) #model_LRS
model_test1.to(device1)

model_test2 = UNet(input_channel2, output_channel) #model_fss
model_test2.to(device2)

#To continue training from the trained model

#TDL_path = '/main/models/DSA_LTD/model0.path'
#model_test0.load_state_dict(torch.load(TDL_path, map_location = {'cuda:1':'cuda:1'}))

#LRS_path = '/main/models/DSA_LTD/model1.path'
#model_test1.load_state_dict(torch.load(LRS_path, map_location = {'cuda:2':'cuda:2'}))

#
#FSS_path = '/main/models/DSA_LTD/model2.path'
#model_test2.load_state_dict(torch.load(FSS_path, map_location = {'cuda:6':'cuda:3'}))


#######################################################
# Passing the Dataset of Images and Labels
#######################################################
data_root = '/main/data'
train_img_data = data_root + '/train/pre_img'
train_liver_data = data_root +'/train/liver_labels_wb'
train_diff_data = data_root +'/train/keydiffer3'
train_imgs_data = data_root +'/train/img10'
train_label_data = data_root +'/train/pre_labels_wb'
Training_Data = Images_Dataset_folder(train_img_data, train_liver_data, train_diff_data, train_imgs_data, train_label_data, 'train')

T_Data = Training_Data
num_train = len(T_Data)
print (num_train)


#====test==
test_img_data = data_root +'/test/pre_img'
test_liver_data = data_root +'/test/liver_labels_wb'
test_diff_data = data_root +'/test/keydiffer3'
test_imgs_data =  data_root +'/test/img10'
test_label_data = data_root + '/test/pre_labels_wb'
Test_Data = Images_Dataset_folder(test_img_data, test_liver_data,test_diff_data, test_imgs_data, test_label_data, 'val')

V_Data = Test_Data
num_val = len(V_Data)
print (num_val)

dataloaders_dict = {
    'train': torch.utils.data.DataLoader(T_Data, batch_size=batch_size, shuffle=True, num_workers=4,
                                         pin_memory=pin_memory),
    'val': torch.utils.data.DataLoader(V_Data, batch_size=1, shuffle=False, num_workers=4,
                                       pin_memory=pin_memory)}

#######################################################
# Using Adam as Optimizer
#######################################################
lr = 1e-4

params =([{"params":model_test2.parameters(), "lr": lr},
          {"params":model_test1.parameters(), "lr": lr/100},
          {"params":model_test0.parameters(), "lr": lr},
         ])

# optimizer = torch.optim.SGD(params, lr = lr, momentum=0.8)

criterion = torch.nn.L1Loss()

initial_lr = 0.001
optimizer = torch.optim.Adam(params, lr=lr)

MAX_STEP = int(1e10)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, MAX_STEP, eta_min=1e-8)
#######################################################
# Writing the params to tensorboard
#######################################################
output_root ='./exp0'

board_dir = os.path.join(output_root, 'board0')
writer1 = SummaryWriter(board_dir)
check_create_dir(board_dir)

output_dir = os.path.join(output_root, 'models')
check_create_dir(output_dir)

#######################################################
# Creating a Folder for every data of the program
#######################################################
model0_dir = os.path.join(output_dir, 'model0')
model1_dir = os.path.join(output_dir, 'model1')
model2_dir = os.path.join(output_dir, 'model2')

check_create_dir(model0_dir)
check_create_dir(model1_dir)
check_create_dir(model2_dir)
#######################################################s
# Training
#######################################################
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
    model_test0.eval()
    model_test1.eval()
    model_test2.train()

    k = 1

    for fm, x, li, y, z, name in dataloaders_dict[phase]:


        optimizer.zero_grad()
        fm, y = fm.to(device), y.to(device2)
        
        x, li = x.to(device1), li.to(device2)
        z = z.to(device2)
        x2 = x.to(device2)


        y_pred = model_test0(fm)
        y_pred = y_pred.to(device2)
        loss_A = criterion(y_pred, y)

      
        li_pred = model_test1(x)
        li_pred = li_pred.to(device2)
        loss_B = calc_loss(li_pred, li)
        
        
        input = torch.cat((x2, li_pred, y_pred), 1)
        input = input.to(device2)
        z_pred = model_test2(input)


        loss_C = calc_loss(z_pred, z)
        

        f_loss = lamda0 * loss_A + lamda1 * loss_B + lamda2 * loss_C

        f_loss.backward()
        optimizer.step()
        
        train_loss += f_loss.item() * x.size(0)
        x_size = f_loss.item() * x.size(0)

        k = 2

    train_losses.append(train_loss)
    scheduler.step()

    #######################################################
    # Validation Step
    #######################################################
    phase = 'val'
    model_test0.eval()
    model_test1.eval()
    model_test2.eval()
    dice = 0
    for fm1, x1, li1, y1, z1, name1 in dataloaders_dict[phase]:
        fm1, y1 = fm1.to(device), y1.to(device2)
        
        x1, li1 = x1.to(device1), li1.to(device2)
        z1 = z1.to(device2)
        x21 = x1.to(device2)


        y_pred1 = model_test0(fm1)
        y_pred1 = y_pred1.to(device2)
        loss_A = criterion(y_pred1, y1)

        li_pred1 = model_test1(x1)
        li_pred1 = li_pred1.to(device2)
        loss_B = calc_loss(li_pred1, li1)
        
        
        input1 = torch.cat((x21, li_pred1, y_pred1), 1)
        input1 = input1.to(device2)
        z_pred1 = model_test2(input1)
        
        loss_C = calc_loss(z_pred1, z1)
        
        f_loss = lamda0 * loss_A + lamda1 * loss_B + lamda2 * loss_C


        valid_loss += f_loss.item() * x1.size(0)
        x1_size = f_loss.item() * x1.size(0)
        
        #evaluate
        z_pred1 = torch.sigmoid(z_pred1)
        shreshold2 = 0.2
        z_pred1[z_pred1 < shreshold2] = 0
        z_pred1[z_pred1 >= shreshold2] = 1
        
        z1 = z1.squeeze(0)
        z_pred1 = z_pred1.squeeze(0)
        z1 = z1.cpu().detach()
        z_pred1 = z_pred1.cpu().detach()
        
        d2 = dice_coeff(z_pred1, z1)
        dice += d2

    val_losses.append(valid_loss)
    dice = dice / num_val
    if i == 0: 
        print (dice)
    #######################################################
    # To write in Tensorboard
    #######################################################

    train_loss = train_loss / num_train
    valid_loss = valid_loss / num_val
    writer1.add_scalars('Train_val_loss', {'train_loss': train_loss}, i)
    writer1.add_scalars('Train_val_loss', {'val_loss': valid_loss}, i)

    if (i + 1) % 1 == 0:
        print('Epoch: {}/{} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(i + 1, epoch, train_loss,
                                                                                      valid_loss))

    #######################################################
    # Early Stopping
    #######################################################

    if dice_max <= dice and epoch_valid >= i:  # and i_valid <= 2:
        print('Validation Dice increased ({:.6f} --> {:.6f}).  Saving model '.format(dice_max, dice))

        model_name = 'dice_max.path'
        model_path0 = os.path.join(model0_dir, model_name)
        torch.save(model_test0.state_dict(), model_path0)
        
        model_path1 = os.path.join(model1_dir, model_name)
        torch.save(model_test1.state_dict(), model_path1)

        model_path2 = os.path.join(model2_dir, model_name)
        torch.save(model_test2.state_dict(), model_path2)

        if round(dice, 4) == round(dice_max, 4):
            print(i_valid)
            i_valid = i_valid + 1
        dice_max = dice
#######################################################
# closing the tensorboard writer
#######################################################

writer1.close()
