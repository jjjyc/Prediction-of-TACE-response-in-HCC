from __future__ import print_function, division
import os
import numpy as np


import torch.utils.data
import torch.nn
import shutil
import random

from Metrics import dice_coeff, accuracy_score, recall_score, specificity_score, ppv_score, npv_score
from unet import UNet
from Data_Loader import Images_Dataset_folder
import torchvision.transforms as transforms
from utils import check_create_dir
np.set_printoptions(threshold=np.inf)
#######################################################
# Setting GPU
#######################################################

device = torch.device("cuda:1")
device1 = torch.device("cuda:2")
device2 = torch.device("cuda:3")

#######################################################
# Setting the basic paramters of the model
#######################################################
# data loader
num_workers = 4
pin_memory = False
# train setting

batch_size = 8
# data info

input_channel0 = 10
input_channel1 = 1
input_channel2 = 3
output_channel = 1
# loss
lamda0 = 0.1
lamda1 = 1
lamda2 = 2

SEED = 25
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

#######################################################
# Setting up the model
#######################################################
model_root = '/main/models/DSA_LTD'

model_test0 = UNet(input_channel0, output_channel)  #model_TDL
TDL_path = model_root + '/model0.path'
model_test0.load_state_dict(torch.load(TDL_path, map_location = {'cuda:1':'cuda:1'}))
model_test0.to(device)

model_test1 = UNet(input_channel1, output_channel) #model_LRS
LRS_path =  model_root + '/model1.path'
model_test1.load_state_dict(torch.load(LRS_path, map_location = {'cuda:2':'cuda:2'}))
model_test1.to(device1)

model_test2 = UNet(input_channel2, output_channel) #model_fss
FSS_path =  model_root + '/model2.path'
model_test2.load_state_dict(torch.load(FSS_path, map_location = {'cuda:3':'cuda:3'}))
model_test2.to(device2)

model_test0.eval()
model_test1.eval()
model_test2.eval()

#######################################################
# Passing the Dataset of Images and Labels
#######################################################


data_root = '/main/data'
val_img_data = data_root + '/test/pre_img'
val_liver_data = data_root + '/test/liver_labels_wb'
val_diff_data = data_root + '/test/keydiffer3'
val_imgs_data = data_root + '/test/img10'
val_label_data = data_root + '/test/pre_labels_wb'

Val_Data = Images_Dataset_folder(val_img_data, val_liver_data, val_diff_data, val_imgs_data, val_label_data, 'val')

test_Data = Val_Data
num_test = len(test_Data)
print (num_test)

dataloaders_dict = torch.utils.data.DataLoader(test_Data, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
#######################################################
# output dir
#######################################################

output_root = './test_result/
check_create_dir(output_root)

dir_pre1 = os.path.join(output_root, 'pred1')
dir_gt1 = os.path.join(output_root, 'gt1')
dir_pre2 = os.path.join(output_root, 'pred2')
dir_gt2 = os.path.join(output_root, 'gt2')

output_dir_list = [dir_pre1, dir_gt1,dir_pre2, dir_gt2]

for x in output_dir_list:
    if os.path.exists(x) and os.path.isdir(x):
        shutil.rmtree(x)
    try:
        os.mkdir(x)
    except OSError:
        print("Creation of the testing directory %s failed" % x)
        exit()
    else:
        print("Successfully created the testing directory %s " % x)

############################################
#bootstrap function
############################################


def average(data):
    return sum(data) / len(data)


def bootstrap(data, B, c, func):
    array = np.array(data)
    n = len(array)
    sample_result_arr = []
    for i in range(B):
        index_arr = np.random.randint(0, n, size=n)
        data_sample = array[index_arr]
        sample_result = func(data_sample)
        sample_result_arr.append(sample_result)

    a = 1 - c
    k1 = int(B * a / 2)
    k2 = int(B * (1 - a / 2))
    auc_sample_arr_sorted = sorted(sample_result_arr)
    lower = auc_sample_arr_sorted[k1]
    higher = auc_sample_arr_sorted[k2]

    return round(lower,4), round(higher,4)


#######################################################
# Test
#######################################################


# recall = 0
count = 0

dice2 = 0
recall = 0
specificity = 0
dice1list = []
dice2list = []
recalllist = []
specificitylist = []
acclist = []
PPVlist = []
NPVlist = []

for i, data in enumerate(dataloaders_dict):
        fm, x, y, li, z, basename = data
        filename = basename[0]

        fm, li = fm.to(device), li.to(device2)
        
        x, y = x.to(device1), y.to(device2)
        z = z.to(device2)
        x2 = x.to(device2)
        
        li_pred = model_test0(fm)
        li_pred = li_pred.to(device2)

        y_pred1 = model_test1(x)

        y_pred2 = y_pred1.to(device2)

        y_pred1 = torch.sigmoid(y_pred1)


        shreshold1 = 0.2
        y_pred1[y_pred1 < shreshold1] = 0
        y_pred1[y_pred1 >= shreshold1] = 1

        y = y.squeeze(0)
        y_pred1 = y_pred1.squeeze(0)
        y = y.cpu().detach()
        y_pred1 = y_pred1.cpu().detach()

        d1 = dice_coeff(y_pred1, y)


        input = torch.cat((x2, y_pred2, li_pred), 1)
        input = input.to(device2)
        z_pred1 = model_test2(input)
        z_pred1 = torch.sigmoid(z_pred1)


        shreshold2 = 0.2
        z_pred1[z_pred1 < shreshold2] = 0
        z_pred1[z_pred1 >= shreshold2] = 1


        z = z.squeeze(0)
        z_pred1 = z_pred1.squeeze(0)
        z = z.cpu().detach()
        z_pred1 = z_pred1.cpu().detach()


        d2 = dice_coeff(z_pred1, z)

        r = recall_score(z_pred1, z)
        sp = specificity_score(z_pred1, z)
        c2 = accuracy_score(z_pred1, z)
        npv = npv_score(z_pred1, z)
        ppv = ppv_score(z_pred1, z)
        
        dice1list.append(d1)
        dice2list.append(d2)
        recalllist.append(r)
        specificitylist.append(sp)
        acclist.append(c2)
        NPVlist.append(npv)
        PPVlist.append(ppv)


        transform = transforms.ToPILImage()
        y = transform(y)
        y_pred1 = transform(y_pred1)
        z = transform(z)
        z_pred1 = transform(z_pred1)

        y = y.convert("L")
        y_pred1 = y_pred1.convert("L")
        z = z.convert("L")
        z_pred1 = z_pred1.convert("L")


        pre_path1 = os.path.join(dir_pre1, filename)
        y_pred1.save(pre_path1)

        gt_path1 = os.path.join(dir_gt1, filename)
        y.save(gt_path1)

        pre_path2 = os.path.join(dir_pre2, filename)
        z_pred1.save(pre_path2)

        gt_path2 = os.path.join(dir_gt2, filename)
        z.save(gt_path2)
        count += 1

print('-----------LRS ----------')
dddddddd = bootstrap(dice1list,1000,0.95,average)
print('dice1:' , dddddddd)
print ('mean dice:', average(np.array(dice1list)))

print('-----------final ----------')
dddddddd = bootstrap(dice2list,1000,0.95,average)
print('dice2:' , dddddddd)
# print (dice2/num_test)
print ('mean dice:', average(np.array(dice2list)))
 
rrrrrrrr = bootstrap(recalllist,1000,0.95,average)
print('sensitivity: ', rrrrrrrr)
print ('mean sensitivity:', average(np.array(recalllist)))
   
aaaaaaaa = bootstrap(acclist,1000,0.95,average)
print('accuracy: ',  aaaaaaaa)
print ('mean accuracy:', average(np.array(acclist)))

ssssssss = bootstrap(specificitylist,1000,0.95,average)
print('specificity: ' , ssssssss)
print ('mean specificity:', average(np.array(specificitylist)))
    
ppppppv =  bootstrap(PPVlist,1000,0.95,average)   
print('ppv: ' , ppppppv)  
print ('mean ppv:', average(np.array(PPVlist)))
    
nnnnnpv =  bootstrap(NPVlist,1000,0.95,average)  
print('npv: ' , nnnnnpv)
print ('mean npv:', average(np.array(NPVlist)))

