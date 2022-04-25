from __future__ import print_function, division
import numpy as np
import torch
import os
import torch.utils.data
from PIL import Image
from utils import Discretization
from torchvision import transforms

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
    ]),
}

def load_combined_xy(img_dir, mask_dir, multi_dir, data, feature_names):
    y = data['label']
    leng = len(y)

    name_lenth = len(feature_names)

    Y = []
    X = []
    img_path = []
    ma_path = []
    mu_path = []
    for i in range(leng):
        Y.append(y[i])
        
        dx = []
        for pp in range(name_lenth):
            x = data[feature_names[pp]]
            x_min = x.min()-0.000001
            x_max = x.max()+0.000001
            xi = Discretization(x[i], feature_names[pp], x_min, x_max)
            dx = dx + xi
        img_path.append(os.path.join(img_dir, data['id'][i]))
        ma_path.append(os.path.join(mask_dir, data['id'][i][0:-7] + 'label.png'))
        mu_path.append(os.path.join(multi_dir, data['id'][i]))

        X.append(dx)

    X = np.array(X, dtype = np.float32)
    input_channels = X.shape[1]
    Y = np.array(Y)
    return img_path, ma_path, mu_path, X, Y, input_channels

class Load_Com_Data(torch.utils.data.Dataset):
    def __init__(self, images_dir, masks_dir, multi_dir, feature_data, label_data, phase):

        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.multi_dir = multi_dir
        self.feature_data = feature_data
        self.label_data = label_data
        self.phase = phase
        
    def __len__(self):
        return len(self.label_data)
        
    def __getitem__(self, i):
        inputi = self.feature_data[i]


        labeli = self.label_data[i]
        
        image = Image.open(self.images_dir[i]).convert('L')
        mask = Image.open(self.masks_dir[i]).convert('L')
        multi = Image.open(self.multi_dir[i]).convert('L')

        transform = data_transforms[self.phase]
        img = transform(image)
        mask = transform(mask)
        multi = transform(multi)
        input_img = torch.cat((img, multi, mask), 0)
        return inputi, input_img, labeli
