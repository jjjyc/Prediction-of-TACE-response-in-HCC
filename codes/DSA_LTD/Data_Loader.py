from __future__ import print_function, division

import os
import random
import numpy as np
from PIL import Image


import torch
import torchvision
import torch.utils.data
from torch.utils.data import Dataset
from torchvision import transforms


size = (256, 256)

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(size),
        # transforms.RandomRotation(5),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        transforms.ToTensor(),

    ]),
    'val': transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ]),
}


class Images_Dataset_folder(torch.utils.data.Dataset):
    def __init__(self, images_dir, liver_dir, diff_dir, frames_dir, labels_dir, phase):
        self.images = sorted(os.listdir(images_dir))
        self.livers = sorted(os.listdir(liver_dir))
        self.diffs = sorted(os.listdir(diff_dir))
        self.labels = sorted(os.listdir(labels_dir))
        
        self.frames_dir = frames_dir

        self.images_dir = images_dir
        self.livers_dir = liver_dir
        self.labels_dir = labels_dir
        self.diff_dir = diff_dir

        self.phase = phase

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        # print(self.images[i])
        transform = data_transforms[self.phase]

        i1 = Image.open(os.path.join(self.images_dir, self.images[i]))
        li1 = Image.open(os.path.join(self.livers_dir,self.livers[i]))
        l1 = Image.open(os.path.join(self.labels_dir, self.labels[i]))
        d1 = Image.open(os.path.join(self.diff_dir, self.diffs[i]))


        i1 = i1.convert("L")
        li1 = li1.convert("L")
        d1 = d1.convert("L")

        l1 = l1.convert("L")

        img = transform(i1)
        liver = transform(li1)
        differ = transform(d1)
        label = transform(l1)

        # input=torch.cat((img,liver),0)
        
        name1= self.images[i]
        

        pid = name1[name1.find('P'):-15]
        frame_path = os.path.join(self.frames_dir, pid)
        frames = sorted(os.listdir(frame_path))
        f = []
        for xxx in frames:
            ffff = Image.open(os.path.join(frame_path, xxx))
            ffff = ffff.convert("L")
            ffffd = transform(ffff)
            f.append(ffffd)
        frames_input = torch.cat((f), 0)



        return frames_input, img, liver, differ, label, self.images[i]
